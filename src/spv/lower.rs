//! SPIR-V to SPIR-T lowering.

use crate::spv::{self, spec};
// FIXME(eddyb) import more to avoid `crate::` everywhere.
use crate::{
    cfg, print, AddrSpace, Attr, AttrSet, Const, ConstCtor, ConstDef, Context, ControlNodeDef,
    ControlNodeKind, ControlRegion, ControlRegionDef, ControlRegionInputDecl, DataInst,
    DataInstDef, DataInstFormDef, DataInstKind, DeclDef, Diag, EntityDefs, EntityList, ExportKey,
    Exportee, Func, FuncDecl, FuncDefBody, FuncParam, FxIndexMap, GlobalVarDecl, GlobalVarDefBody,
    Import, InternedStr, Module, SelectionKind, Type, TypeCtor, TypeCtorArg, TypeDef, Value,
};
use itertools::Either;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;
use std::{io, mem};

/// SPIR-T definition of a SPIR-V ID.
enum IdDef {
    Type(Type),
    Const(Const),

    /// Like `Const`, but for SPIR-V "aggregate" (`OpTypeStruct`/`OpTypeArray`)
    /// constants (e.g. `OpConstantComposite`s of those types, but also more
    /// general constants like `OpUndef`/`OpConstantNull` etc.).
    AggregateConst {
        // FIXME(eddyb) remove `whole_const` by always using the `leaves`.
        whole_const: Const,

        leaves: SmallVec<[Const; 2]>,
    },

    Func(Func),

    SpvExtInstImport(InternedStr),
    SpvDebugString(InternedStr),
}

impl IdDef {
    fn descr(&self, cx: &Context) -> String {
        match *self {
            // FIXME(eddyb) print these with some kind of "maximum depth",
            // instead of just describing the kind of definition.
            // FIXME(eddyb) replace these with the `Diag` embedding system.
            IdDef::Type(_) => "a type".into(),
            IdDef::Const(_) => "a constant".into(),
            IdDef::AggregateConst { .. } => "an aggregate constant".into(),

            IdDef::Func(_) => "a function".into(),

            IdDef::SpvExtInstImport(name) => {
                format!("`OpExtInstImport {:?}`", &cx[name])
            }
            IdDef::SpvDebugString(s) => format!("`OpString {:?}`", &cx[s]),
        }
    }
}

/// For SPIR-V types used as (SSA) value types, this specifies how to handle them.
#[derive(Clone)]
enum TypeValueCategory {
    /// SPIR-T [`Value`]s can have this type (see [`Value`] documentation).
    Direct,

    /// This is a SPIR-V "aggregate" type (`OpTypeStruct`/`OpTypeArray`), and
    /// each of its values will be decomposed into its "leaf" [`Value`]s, with
    /// leaves' types being given by `leaf_types`.
    Aggregate { leaf_types: Rc<Vec<Type>> },
}

impl TypeValueCategory {
    fn into_disaggregated_leaf_types(
        self,
        original_type: Type,
    ) -> impl ExactSizeIterator<Item = Type> + Clone {
        match self {
            Self::Direct => Either::Left([original_type].into_iter()),
            Self::Aggregate { leaf_types } => {
                // HACK(eddyb) this avoids having to borrow a `TypeValueCategory`.
                Either::Right((0..leaf_types.len()).map(move |i| leaf_types[i]))
            }
        }
    }
}

/// Error type for when a SPIR-V type cannot have a `TypeValueCategory`, i.e.
/// this type can only be used behind a pointer. Disaggregation won't be
/// performed, so illegal attempts at constructing values of this type will
/// be kept intact, but annotated with an error [`Diag`]nostic.
//
// FIXME(eddyb) include an actual error message in here, maybe a whole `Diag`.
#[derive(Clone)]
struct TypeIsIndirectOnly;

/// Deferred export, needed because the IDs are initially forward refs.
enum Export {
    Linkage {
        name: InternedStr,
        target_id: spv::Id,
    },
    EntryPoint {
        func_id: spv::Id,
        imms: SmallVec<[spv::Imm; 2]>,
        interface_ids: SmallVec<[spv::Id; 4]>,
    },
}

/// Deferred [`FuncDefBody`], needed because some IDs are initially forward refs.
struct FuncBody {
    func_id: spv::Id,
    func: Func,
    insts: Vec<IntraFuncInst>,
}

struct IntraFuncInst {
    // Instruction aspects that can be pre-lowered:
    attrs: AttrSet,
    result_type_and_val_cat: Option<(Type, TypeValueCategory)>,

    without_ids: spv::Inst,

    // Instruction aspects that cannot be lowered initially (due to forward refs):
    result_id: Option<spv::Id>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    ids: SmallVec<[spv::Id; 4]>,
}

// FIXME(eddyb) stop abusing `io::Error` for error reporting and switch to `Diag`.
fn invalid(reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("malformed SPIR-V ({reason})"),
    )
}

// FIXME(eddyb) there should be just one `Lowerer` type with all the state.
struct TypeCategorizer {
    cx: Rc<Context>,
    wk: &'static spec::WellKnown,

    cached_type_value_category: FxHashMap<Type, Result<TypeValueCategory, TypeIsIndirectOnly>>,
}

impl TypeCategorizer {
    fn new(cx: Rc<Context>) -> Self {
        Self {
            cx,
            wk: &spec::Spec::get().well_known,

            cached_type_value_category: FxHashMap::default(),
        }
    }

    fn type_value_category(&mut self, ty: Type) -> Result<TypeValueCategory, TypeIsIndirectOnly> {
        if let Some(cached) = self.cached_type_value_category.get(&ty).cloned() {
            return cached;
        }

        let result = self.uncached_type_value_category(ty);
        self.cached_type_value_category.insert(ty, result.clone());
        result
    }

    fn uncached_type_value_category(
        &mut self,
        ty: Type,
    ) -> Result<TypeValueCategory, TypeIsIndirectOnly> {
        let cx = self.cx.clone();
        let ty_def = &cx[ty];

        let opcode = match &ty_def.ctor {
            TypeCtor::SpvInst(spv_inst) => spv_inst.opcode,
            TypeCtor::QPtr | TypeCtor::SpvStringLiteralForExtInst => {
                return Ok(TypeValueCategory::Direct);
            }
        };

        if opcode == self.wk.OpTypeStruct {
            let mut leaf_types = vec![];
            for &arg in &ty_def.ctor_args {
                let field_type = match arg {
                    TypeCtorArg::Type(ty) => ty,
                    TypeCtorArg::Const(_) => return Err(TypeIsIndirectOnly),
                };
                leaf_types.extend(
                    self.type_value_category(field_type)?
                        .into_disaggregated_leaf_types(field_type),
                );
            }
            Ok(TypeValueCategory::Aggregate {
                leaf_types: Rc::new(leaf_types),
            })
        } else if opcode == self.wk.OpTypeArray {
            let (elem_type, len) = match ty_def.ctor_args[..] {
                [TypeCtorArg::Type(elem_type), TypeCtorArg::Const(len)] => (elem_type, len),
                _ => return Err(TypeIsIndirectOnly),
            };

            // NOTE(eddyb) this can legally be `None` when the length of
            // the array is given by a specialization constant.
            let fixed_len = match &cx[len].ctor {
                ConstCtor::SpvInst(spv_inst)
                    if spv_inst.opcode == self.wk.OpConstant && spv_inst.imms.len() == 1 =>
                {
                    match spv_inst.imms[..] {
                        [spv::Imm::Short(_, x)] => Some(x),
                        _ => unreachable!(),
                    }
                }
                _ => None,
            };

            let elem_leaf_types = self
                .type_value_category(elem_type)?
                .into_disaggregated_leaf_types(elem_type);
            let fixed_len = fixed_len.ok_or(TypeIsIndirectOnly)?;

            let mut leaf_types =
                Vec::with_capacity(elem_leaf_types.len().saturating_mul(fixed_len as usize));
            for _ in 0..fixed_len {
                leaf_types.extend(elem_leaf_types.clone());
            }
            Ok(TypeValueCategory::Aggregate {
                leaf_types: Rc::new(leaf_types),
            })
        } else if opcode == self.wk.OpTypeRuntimeArray {
            Err(TypeIsIndirectOnly)
        } else {
            Ok(TypeValueCategory::Direct)
        }
    }
}

// FIXME(eddyb) provide more information about any normalization that happened:
// * stats about deduplication that occured through interning
// * sets of unused global vars and functions (and types+consts only they use)
// FIXME(eddyb) use `Diag` instead of `io::Error`, maybe with a return type like
// `Result<Module, IncompletelyLoweredModule>` where `IncompletelyLoweredModule`
// contains a `Module`, maps of all the SPIR-V IDs (to the SPIR-T definitions),
// global `Diag`s (where they can't be attached to specific `AttrSet`s), etc.
impl Module {
    pub fn lower_from_spv_file(cx: Rc<Context>, path: impl AsRef<Path>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(cx, spv::read::ModuleParser::read_from_spv_file(path)?)
    }

    pub fn lower_from_spv_bytes(cx: Rc<Context>, spv_bytes: Vec<u8>) -> io::Result<Self> {
        Self::lower_from_spv_module_parser(
            cx,
            spv::read::ModuleParser::read_from_spv_bytes(spv_bytes)?,
        )
    }

    pub fn lower_from_spv_module_parser(
        cx: Rc<Context>,
        parser: spv::read::ModuleParser,
    ) -> io::Result<Self> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        // HACK(eddyb) used to quickly check whether an `OpVariable` is global.
        let storage_class_function_imm = spv::Imm::Short(wk.StorageClass, wk.Function);

        let mut module = {
            let [
                magic,
                version,
                generator_magic,
                id_bound,
                reserved_inst_schema,
            ] = parser.header;

            // Ensured above (this is the value after any endianness swapping).
            assert_eq!(magic, spv_spec.magic);

            let [
                version_reserved_hi,
                version_major,
                version_minor,
                version_reserved_lo,
            ] = version.to_be_bytes();

            if (version_reserved_lo, version_reserved_hi) != (0, 0) {
                return Err(invalid(&format!(
                    "version 0x{version:08x} is not in expected (0.major.minor.0) form"
                )));
            }

            // FIXME(eddyb) maybe use this somehow? (e.g. check IDs against it)
            let _ = id_bound;

            if reserved_inst_schema != 0 {
                return Err(invalid(&format!(
                    "unknown instruction schema {reserved_inst_schema} - only 0 is supported"
                )));
            }

            Self::new(
                cx.clone(),
                crate::ModuleDialect::Spv(spv::Dialect {
                    version_major,
                    version_minor,

                    capabilities: BTreeSet::new(),
                    extensions: BTreeSet::new(),

                    addressing_model: 0,
                    memory_model: 0,
                }),
                crate::ModuleDebugInfo::Spv(spv::ModuleDebugInfo {
                    original_generator_magic: NonZeroU32::new(generator_magic),

                    source_languages: BTreeMap::new(),
                    source_extensions: vec![],
                    module_processes: vec![],
                }),
            )
        };

        #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
        enum Seq {
            Capability,
            Extension,
            ExtInstImport,
            MemoryModel,
            EntryPoint,
            ExecutionMode,
            DebugStringAndSource,
            DebugName,
            DebugModuleProcessed,
            Decoration,

            // NOTE(eddyb) not its own section, but only a "checkpoint", forcing
            // instructions following `OpLine`/`OpNoLine` into later sections.
            DebugLine,

            TypeConstOrGlobalVar,
            Function,
        }
        let mut seq = None;

        // FIXME(eddyb) there should be just one `Lowerer` type with all the state.
        let mut type_categorizer = TypeCategorizer::new(cx.clone());

        let mut has_memory_model = false;
        let mut pending_attrs = FxHashMap::<spv::Id, crate::AttrSetDef>::default();
        let mut pending_imports = FxHashMap::<spv::Id, Import>::default();
        let mut pending_exports = vec![];
        let mut current_debug_line = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_debug_line` resets.
        let mut id_defs = FxHashMap::default();
        let mut pending_func_bodies = vec![];
        let mut current_func_body = None;

        let mut spv_insts = parser.peekable();
        while let Some(mut inst) = spv_insts.next().transpose()? {
            let opcode = inst.opcode;

            let invalid = |msg: &str| invalid(&format!("in {}: {}", opcode.name(), msg));

            // Handle line debuginfo early, as it doesn't have its own section,
            // but rather can go almost anywhere among globals and functions.
            if [wk.OpLine, wk.OpNoLine].contains(&opcode) {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                current_debug_line = if opcode == wk.OpLine {
                    match (&inst.imms[..], &inst.ids[..]) {
                        (
                            &[spv::Imm::Short(l_kind, line), spv::Imm::Short(c_kind, col)],
                            &[file_path_id],
                        ) => {
                            assert_eq!([l_kind, c_kind], [wk.LiteralInteger; 2]);
                            let file_path = match id_defs.get(&file_path_id) {
                                Some(&IdDef::SpvDebugString(s)) => s,
                                _ => {
                                    return Err(invalid(&format!(
                                        "%{file_path_id} is not an OpString"
                                    )));
                                }
                            };
                            Some((file_path, line, col))
                        }
                        _ => unreachable!(),
                    }
                } else {
                    assert!(inst.imms.is_empty() && inst.ids.is_empty());
                    None
                };

                // Advance to `Seq::DebugLine` if we're not there yet, forcing
                // any following instructions to not be in earlier sections.
                seq = seq.max(Some(Seq::DebugLine));
                continue;
            }

            // Reset line debuginfo when crossing/leaving blocks.
            let new_block_id = if opcode == wk.OpLabel {
                Some(inst.result_id.unwrap())
            } else if opcode == wk.OpFunctionEnd {
                None
            } else {
                current_block_id
            };
            if current_block_id != new_block_id {
                current_debug_line = None;
            }
            current_block_id = new_block_id;

            let mut attrs = inst
                .result_id
                .and_then(|id| pending_attrs.remove(&id))
                .unwrap_or_default();

            if let Some((file_path, line, col)) = current_debug_line {
                // FIXME(eddyb) use `get_or_insert_default` once that's stabilized.
                attrs.attrs.insert(Attr::SpvDebugLine {
                    file_path: crate::OrdAssertEq(file_path),
                    line,
                    col,
                });
            }

            // Take certain bitflags operands out of the instruction and rewrite
            // them into attributes instead.
            inst.imms.retain(|imm| match *imm {
                spv::Imm::Short(kind, word) if kind == wk.FunctionControl => {
                    if word != 0 {
                        attrs.attrs.insert(Attr::SpvBitflagsOperand(*imm));
                    }
                    false
                }
                _ => true,
            });

            let mut attrs = cx.intern(attrs);

            // FIXME(eddyb) move this kind of lookup into methods on some sort
            // of "lowering context" type.
            let result_type_and_val_cat = inst
                .result_type_id
                .map(|type_id| match id_defs.get(&type_id) {
                    Some(&IdDef::Type(ty)) => Ok((ty, type_categorizer.type_value_category(ty))),
                    Some(id_def) => Err(invalid(&format!(
                        "result type %{} should be a type, not a {}",
                        type_id,
                        id_def.descr(&cx)
                    ))),
                    None => Err(invalid(&format!("result type %{type_id} not defined"))),
                })
                .transpose()?;

            let inst_category = spv_spec.instructions[opcode].category;

            let next_seq = if opcode == wk.OpCapability {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let cap = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(kind, cap)], &[]) => {
                        assert_eq!(kind, wk.Capability);
                        cap
                    }
                    _ => unreachable!(),
                };

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.capabilities.insert(cap);
                    }
                }

                Seq::Capability
            } else if opcode == wk.OpExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.extensions.insert(ext);
                    }
                }

                Seq::Extension
            } else if opcode == wk.OpExtInstImport {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let name = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvExtInstImport(cx.intern(name)));

                Seq::ExtInstImport
            } else if opcode == wk.OpMemoryModel {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (addressing_model, memory_model) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[spv::Imm::Short(am_kind, am), spv::Imm::Short(mm_kind, mm)], &[]) => {
                        assert_eq!([am_kind, mm_kind], [wk.AddressingModel, wk.MemoryModel]);
                        (am, mm)
                    }
                    _ => unreachable!(),
                };

                if has_memory_model {
                    return Err(invalid("duplicate OpMemoryModel"));
                }
                has_memory_model = true;

                match &mut module.dialect {
                    crate::ModuleDialect::Spv(dialect) => {
                        dialect.addressing_model = addressing_model;
                        dialect.memory_model = memory_model;
                    }
                }

                Seq::MemoryModel
            } else if opcode == wk.OpString {
                assert!(inst.result_type_id.is_none() && inst.ids.is_empty());
                let id = inst.result_id.unwrap();
                let s = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                id_defs.insert(id, IdDef::SpvDebugString(cx.intern(s)));

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSource {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (lang, version) = match inst.imms[..] {
                    [
                        spv::Imm::Short(l_kind, lang),
                        spv::Imm::Short(v_kind, version),
                        ..,
                    ] => {
                        assert_eq!([l_kind, v_kind], [wk.SourceLanguage, wk.LiteralInteger]);
                        (lang, version)
                    }
                    _ => unreachable!(),
                };

                let debug_sources = match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => debug_info
                        .source_languages
                        .entry(spv::DebugSourceLang { lang, version })
                        .or_default(),
                };

                match (&inst.imms[2..], &inst.ids[..]) {
                    (contents, &[file_path_id]) => {
                        let file_path = match id_defs.get(&file_path_id) {
                            Some(&IdDef::SpvDebugString(s)) => s,
                            _ => {
                                return Err(invalid(&format!(
                                    "%{file_path_id} is not an OpString"
                                )));
                            }
                        };
                        let mut contents = if contents.is_empty() {
                            String::new()
                        } else {
                            spv::extract_literal_string(contents)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?
                        };

                        // Absorb all following `OpSourceContinued` into `contents`.
                        while let Some(Ok(cont_inst)) = spv_insts.peek() {
                            if cont_inst.opcode != wk.OpSourceContinued {
                                break;
                            }
                            let cont_inst = spv_insts.next().unwrap().unwrap();

                            assert!(
                                cont_inst.result_type_id.is_none()
                                    && cont_inst.result_id.is_none()
                                    && cont_inst.ids.is_empty()
                            );
                            let cont_contents = spv::extract_literal_string(&cont_inst.imms)
                                .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                            contents += &cont_contents;
                        }

                        debug_sources.file_contents.insert(file_path, contents);
                    }
                    (&[], &[]) => {}
                    _ => unreachable!(),
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpSourceContinued {
                return Err(invalid("must follow OpSource"));
            } else if opcode == wk.OpSourceExtension {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let ext = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.source_extensions.push(ext);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugStringAndSource
            } else if opcode == wk.OpModuleProcessed {
                assert!(
                    inst.result_type_id.is_none()
                        && inst.result_id.is_none()
                        && inst.ids.is_empty()
                );
                let proc = spv::extract_literal_string(&inst.imms)
                    .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;

                match &mut module.debug_info {
                    crate::ModuleDebugInfo::Spv(debug_info) => {
                        debug_info.module_processes.push(proc);
                    }
                }

                // NOTE(eddyb) debug instructions are handled earlier in the code
                // for organizatory purposes, see `Seq` for the in-module order.
                Seq::DebugModuleProcessed
            } else if opcode == wk.OpEntryPoint {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                pending_exports.push(Export::EntryPoint {
                    func_id: inst.ids[0],
                    imms: inst.without_ids.imms,
                    interface_ids: inst.ids[1..].iter().copied().collect(),
                });

                Seq::EntryPoint
            } else if [
                wk.OpExecutionMode,
                wk.OpExecutionModeId, // FIXME(eddyb) not actually supported
                wk.OpName,
                wk.OpMemberName,
                wk.OpDecorate,
                wk.OpMemberDecorate,
                wk.OpDecorateId, // FIXME(eddyb) not actually supported
                wk.OpDecorateString,
                wk.OpMemberDecorateString,
            ]
            .contains(&opcode)
            {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());

                let target_id = inst.ids[0];
                if inst.ids.len() > 1 {
                    return Err(invalid("unsupported decoration with ID"));
                }

                match inst.imms[..] {
                    // Special-case `OpDecorate LinkageAttributes ... Import|Export`.
                    [
                        decoration @ spv::Imm::Short(..),
                        ref name @ ..,
                        spv::Imm::Short(lt_kind, linkage_type),
                    ] if opcode == wk.OpDecorate
                        && decoration == spv::Imm::Short(wk.Decoration, wk.LinkageAttributes)
                        && lt_kind == wk.LinkageType
                        && [wk.Import, wk.Export].contains(&linkage_type) =>
                    {
                        let name = spv::extract_literal_string(name)
                            .map_err(|e| invalid(&format!("{} in {:?}", e, e.as_bytes())))?;
                        let name = cx.intern(name);

                        if linkage_type == wk.Import {
                            pending_imports.insert(target_id, Import::LinkName(name));
                        } else {
                            pending_exports.push(Export::Linkage { name, target_id });
                        }
                    }

                    _ => {
                        pending_attrs
                            .entry(target_id)
                            .or_default()
                            .attrs
                            .insert(Attr::SpvAnnotation(inst.without_ids));
                    }
                };

                if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(&opcode) {
                    Seq::ExecutionMode
                } else if [wk.OpName, wk.OpMemberName].contains(&opcode) {
                    Seq::DebugName
                } else {
                    Seq::Decoration
                }
            } else if [
                wk.OpDecorationGroup,
                wk.OpGroupDecorate,
                wk.OpGroupMemberDecorate,
            ]
            .contains(&opcode)
            {
                return Err(invalid(
                    "unsupported decoration groups (officially deprecated)",
                ));
            } else if opcode == wk.OpTypeForwardPointer {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                let (id, sc) = match (&inst.imms[..], &inst.ids[..]) {
                    (&[sc], &[id]) => (id, sc),
                    _ => unreachable!(),
                };

                // HACK(eddyb) this is not a proper implementation - one would
                // require fixpoint (aka "μ" aka "mu") types - but for now this
                // serves as a first approximation for a "deferred error".
                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    ctor: TypeCtor::SpvInst(spv::Inst {
                        opcode,
                        imms: [sc].into_iter().collect(),
                    }),
                    ctor_args: [].into_iter().collect(),
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Type {
                assert!(inst.result_type_id.is_none());
                let id = inst.result_id.unwrap();
                let type_ctor_args = inst
                    .ids
                    .iter()
                    .map(|&id| match id_defs.get(&id) {
                        Some(&IdDef::Type(ty)) => Ok(TypeCtorArg::Type(ty)),
                        Some(&IdDef::Const(ct)) => Ok(TypeCtorArg::Const(ct)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .map(|result| {
                        result.map_err(|descr| {
                            invalid(&format!("unsupported use of {descr} in a type"))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                let ty = cx.intern(TypeDef {
                    attrs: mem::take(&mut attrs),
                    ctor: TypeCtor::SpvInst(inst.without_ids),
                    ctor_args: type_ctor_args,
                });
                id_defs.insert(id, IdDef::Type(ty));

                Seq::TypeConstOrGlobalVar
            } else if inst_category == spec::InstructionCategory::Const || opcode == wk.OpUndef {
                let id = inst.result_id.unwrap();

                let (ty, ty_val_cat) = result_type_and_val_cat.unwrap();

                let mut aggregate_leaves = match &ty_val_cat {
                    Ok(TypeValueCategory::Direct) => None,
                    Err(TypeIsIndirectOnly) => {
                        attrs.push_diag(
                            &cx,
                            Diag::err([
                                "values of type `".into(),
                                ty.into(),
                                "` can only be accessed through pointers".into(),
                            ]),
                        );
                        None
                    }
                    Ok(TypeValueCategory::Aggregate { leaf_types }) => {
                        if [wk.OpConstantNull, wk.OpUndef].contains(&opcode) {
                            assert_eq!(inst.ids.len(), 0);
                            Some(
                                leaf_types
                                    .iter()
                                    .map(|&leaf_type| {
                                        cx.intern(ConstDef {
                                            attrs: Default::default(),
                                            ty: leaf_type,
                                            ctor: ConstCtor::SpvInst(opcode.into()),
                                            ctor_args: [].into_iter().collect(),
                                        })
                                    })
                                    .collect(),
                            )
                        } else if [wk.OpConstantComposite, wk.OpSpecConstantComposite]
                            .contains(&opcode)
                        {
                            // NOTE(eddyb) actual leaves gathered below, while
                            // collecting `const_ctor_args`.
                            Some(SmallVec::with_capacity(leaf_types.len()))
                        } else {
                            attrs.push_diag(
                                &cx,
                                Diag::bug(["unsupported aggregate-producing constant".into()]),
                            );
                            None
                        }
                    }
                };

                let const_ctor_args = inst
                    .ids
                    .iter()
                    .map(|&id| match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => {
                            if let Some(aggregate_leaves) = &mut aggregate_leaves {
                                aggregate_leaves.push(ct);
                            }
                            Ok(ct)
                        }
                        Some(IdDef::AggregateConst {
                            whole_const,
                            leaves,
                        }) => {
                            if let Some(aggregate_leaves) = &mut aggregate_leaves {
                                aggregate_leaves.extend(leaves.iter().copied());
                            }
                            Ok(*whole_const)
                        }
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .map(|result| {
                        result.map_err(|descr| {
                            invalid(&format!("unsupported use of {descr} in a constant"))
                        })
                    })
                    .collect::<Result<_, _>>()?;

                if let (Ok(TypeValueCategory::Aggregate { leaf_types }), Some(leaves)) =
                    (&ty_val_cat, &aggregate_leaves)
                {
                    if leaf_types.len() != leaves.len() {
                        attrs.push_diag(
                            &cx,
                            Diag::err([format!(
                                "aggregate leaf count mismatch (expected {}, found {})",
                                leaf_types.len(),
                                leaves.len()
                            )
                            .into()]),
                        );
                        aggregate_leaves = None;
                    }
                }

                let ct = cx.intern(ConstDef {
                    attrs: mem::take(&mut attrs),
                    ty,
                    ctor: ConstCtor::SpvInst(inst.without_ids),
                    ctor_args: const_ctor_args,
                });
                id_defs.insert(
                    id,
                    match (ty_val_cat, aggregate_leaves) {
                        (Ok(TypeValueCategory::Aggregate { .. }), Some(leaves)) => {
                            // FIXME(eddyb) this may lose semantic `attrs` when
                            // `leaves` are directly used.
                            IdDef::AggregateConst {
                                whole_const: ct,
                                leaves,
                            }
                        }
                        _ => IdDef::Const(ct),
                    },
                );

                if opcode == wk.OpUndef {
                    // `OpUndef` can appear either among constants, or in a
                    // function, so at most advance `seq` to globals.
                    seq.max(Some(Seq::TypeConstOrGlobalVar)).unwrap()
                } else {
                    Seq::TypeConstOrGlobalVar
                }
            } else if opcode == wk.OpVariable && current_func_body.is_none() {
                let global_var_id = inst.result_id.unwrap();
                let (type_of_ptr_to_global_var, _) = result_type_and_val_cat.unwrap();

                if inst.imms[0] == storage_class_function_imm {
                    return Err(invalid("`Function` storage class outside function"));
                }

                let storage_class = match inst.imms[..] {
                    [spv::Imm::Short(kind, storage_class)] => {
                        assert_eq!(kind, wk.StorageClass);
                        storage_class
                    }
                    _ => unreachable!(),
                };
                let initializer = match inst.ids[..] {
                    [initializer] => Some(initializer),
                    [] => None,
                    _ => unreachable!(),
                };

                let initializer = initializer
                    .map(|id| match id_defs.get(&id) {
                        Some(&IdDef::Const(ct)) => Ok(ct),
                        Some(&IdDef::AggregateConst { whole_const, .. }) => {
                            // FIXME(eddyb) disaggregate global initializers.
                            Ok(whole_const)
                        }
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("a forward reference to %{id}")),
                    })
                    .transpose()
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the initializer of a global variable"
                        ))
                    })?;

                let def = match pending_imports.remove(&global_var_id) {
                    Some(import @ Import::LinkName(name)) => {
                        if initializer.is_some() {
                            return Err(invalid(&format!(
                                "global variable with initializer decorated as `Import` of {:?}",
                                &cx[name]
                            )));
                        }
                        DeclDef::Imported(import)
                    }
                    None => DeclDef::Present(GlobalVarDefBody { initializer }),
                };

                let global_var = module.global_vars.define(
                    &cx,
                    GlobalVarDecl {
                        attrs: mem::take(&mut attrs),
                        type_of_ptr_to: type_of_ptr_to_global_var,
                        shape: None,
                        addr_space: AddrSpace::SpvStorageClass(storage_class),
                        def,
                    },
                );
                let ptr_to_global_var = cx.intern(ConstDef {
                    attrs: AttrSet::default(),
                    ty: type_of_ptr_to_global_var,
                    ctor: ConstCtor::PtrToGlobalVar(global_var),
                    ctor_args: [].into_iter().collect(),
                });
                id_defs.insert(global_var_id, IdDef::Const(ptr_to_global_var));

                Seq::TypeConstOrGlobalVar
            } else if opcode == wk.OpFunction {
                if current_func_body.is_some() {
                    return Err(invalid("nested OpFunction while still in a function"));
                }

                let func_id = inst.result_id.unwrap();
                let (func_ret_type, func_ret_type_val_cat) = result_type_and_val_cat.unwrap();

                let func_type_id = match (&inst.imms[..], &inst.ids[..]) {
                    // NOTE(eddyb) the `FunctionControl` operand is already gone,
                    // having been converted into an attribute above.
                    (&[], &[func_type_id]) => func_type_id,
                    _ => unreachable!(),
                };

                let (func_type_ret_type, func_type_param_types) =
                    match id_defs.get(&func_type_id) {
                        Some(&IdDef::Type(ty)) => {
                            let ty_def = &cx[ty];
                            match &ty_def.ctor {
                                // FIXME(eddyb) TODO!!! flatten both params and return!
                                TypeCtor::SpvInst(inst) if inst.opcode == wk.OpTypeFunction => {
                                    let mut types = ty_def.ctor_args.iter().map(|&arg| match arg {
                                        TypeCtorArg::Type(ty) => ty,
                                        TypeCtorArg::Const(_) => unreachable!(),
                                    });
                                    Some((types.next().unwrap(), types))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    }
                    .ok_or_else(|| {
                        invalid(&format!(
                            "function type %{func_type_id} not an `OpTypeFunction`"
                        ))
                    })?;

                if func_ret_type != func_type_ret_type {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{}, return type differs between `OpFunction` (expected) \
                         and `OpTypeFunction` (found):\n\n{}",
                        func_id,
                        print::Plan::for_root(
                            &cx,
                            &print::ExpectedVsFound {
                                expected: func_ret_type,
                                found: func_type_ret_type,
                            }
                        )
                        .pretty_print()
                    )));
                }

                let def = match pending_imports.remove(&func_id) {
                    Some(import) => DeclDef::Imported(import),
                    None => {
                        let mut control_regions = EntityDefs::default();
                        let body = control_regions.define(
                            &cx,
                            ControlRegionDef {
                                inputs: SmallVec::new(),
                                children: EntityList::empty(),
                                outputs: SmallVec::new(),
                            },
                        );
                        DeclDef::Present(FuncDefBody {
                            control_regions,
                            control_nodes: Default::default(),
                            data_insts: Default::default(),
                            body,
                            unstructured_cfg: Some(cfg::ControlFlowGraph::default()),
                        })
                    }
                };

                // Always flatten aggregates in param and return types.
                let ret_disaggregated_leaf_types = match &cx[func_ret_type].ctor {
                    // HACK(eddyb) `OpTypeVoid` special-cased here as if it were
                    // an aggregate with `0` leaves.
                    TypeCtor::SpvInst(func_ret_type_spv_inst)
                        if func_ret_type_spv_inst.opcode == wk.OpTypeVoid =>
                    {
                        Either::Left([].into_iter())
                    }

                    // FIXME(eddyb) attach a `Diag` to the function when `Err`.
                    _ => Either::Right(
                        func_ret_type_val_cat
                            .unwrap_or(TypeValueCategory::Direct)
                            .into_disaggregated_leaf_types(func_ret_type),
                    ),
                };
                let params_disaggregated_leaf_types =
                    func_type_param_types.flat_map(|param_type| {
                        // FIXME(eddyb) attach a `Diag` to the function when `Err`.
                        type_categorizer
                            .type_value_category(param_type)
                            .unwrap_or(TypeValueCategory::Direct)
                            .into_disaggregated_leaf_types(param_type)
                    });

                let func = module.funcs.define(
                    &cx,
                    FuncDecl {
                        attrs: mem::take(&mut attrs),
                        ret_types: ret_disaggregated_leaf_types.collect(),
                        params: params_disaggregated_leaf_types
                            .map(|ty| FuncParam {
                                attrs: AttrSet::default(),
                                ty,
                            })
                            .collect(),
                        def,
                    },
                );
                id_defs.insert(func_id, IdDef::Func(func));

                current_func_body = Some(FuncBody {
                    func_id,
                    func,
                    insts: vec![],
                });

                Seq::Function
            } else if opcode == wk.OpFunctionEnd {
                assert!(inst.result_type_id.is_none() && inst.result_id.is_none());
                assert!(inst.imms.is_empty() && inst.ids.is_empty());

                let func_body = current_func_body
                    .take()
                    .ok_or_else(|| invalid("nested OpFunction while still in a function"))?;

                pending_func_bodies.push(func_body);

                Seq::Function
            } else {
                let func_body = current_func_body
                    .as_mut()
                    .ok_or_else(|| invalid("expected only inside a function"))?;
                assert_eq!(seq, Some(Seq::Function));

                let result_type_and_val_cat = result_type_and_val_cat.map(|(ty, ty_val_cat)| {
                    (
                        ty,
                        ty_val_cat.unwrap_or_else(|TypeIsIndirectOnly| {
                            attrs.push_diag(
                                &cx,
                                Diag::err([
                                    "values of type `".into(),
                                    ty.into(),
                                    "` can only be accessed through pointers".into(),
                                ]),
                            );
                            TypeValueCategory::Direct
                        }),
                    )
                });

                func_body.insts.push(IntraFuncInst {
                    attrs: mem::take(&mut attrs),
                    result_type_and_val_cat,

                    without_ids: spv::Inst {
                        opcode,
                        imms: inst.without_ids.imms,
                    },
                    result_id: inst.result_id,
                    ids: inst.ids,
                });

                Seq::Function
            };
            if let Some(prev_seq) = seq {
                if prev_seq > next_seq {
                    return Err(invalid(&format!(
                        "out of order: {next_seq:?} instructions must precede {prev_seq:?} instructions"
                    )));
                }
            }
            seq = Some(next_seq);

            if attrs != Default::default() {
                return Err(invalid("unused decorations / line debuginfo"));
            }
        }

        if !has_memory_model {
            return Err(invalid("missing OpMemoryModel"));
        }

        if !pending_attrs.is_empty() {
            let ids = pending_attrs.keys().collect::<BTreeSet<_>>();
            return Err(invalid(&format!("decorated IDs never defined: {ids:?}")));
        }

        if current_func_body.is_some() {
            return Err(invalid("OpFunction without matching OpFunctionEnd"));
        }

        // HACK(eddyb) `OpNop` is useful for defining `DataInst`s before they're
        // actually lowered (to be able to refer to their outputs `Value`s).
        let mut cached_op_nop_form = None;
        let mut get_op_nop_form = || {
            *cached_op_nop_form.get_or_insert_with(|| {
                cx.intern(DataInstFormDef {
                    kind: DataInstKind::SpvInst(wk.OpNop.into(), Default::default()),
                    output_types: Default::default(),
                })
            })
        };

        // Process function bodies, having seen the whole module.
        for func_body in pending_func_bodies {
            let FuncBody {
                func_id,
                func,
                insts: raw_insts,
            } = func_body;

            let func_decl = &mut module.funcs[func];

            // HACK(eddyb) this is generic to allow `IdDef::AggregateConst`s
            // to be converted to `LocalIdDef::AggregateValue`s, inside
            // `lookup_global_or_local_id_for_data_or_control_inst_input`.
            #[derive(Clone)]
            enum LocalIdDef<AVL = ValueRange> {
                Value(Value),
                AggregateValue { ty: Type, leaves: AVL },
                BlockLabel(ControlRegion),
            }

            #[derive(Clone)]
            struct ValueRange {
                region_or_data_inst: Either<ControlRegion, DataInst>,
                range: Range<u32>,
            }

            impl ValueRange {
                fn iter(&self) -> impl ExactSizeIterator<Item = Value> {
                    let region_or_data_inst = self.region_or_data_inst;
                    self.range.clone().map(move |i| match region_or_data_inst {
                        Either::Left(region) => Value::ControlRegionInput {
                            region,
                            input_idx: i,
                        },
                        Either::Right(inst) => Value::DataInstOutput {
                            inst,
                            output_idx: i,
                        },
                    })
                }
            }

            #[derive(PartialEq, Eq, Hash)]
            struct PhiKey {
                source_block_id: spv::Id,
                target_block_id: spv::Id,
                // FIXME(eddyb) remove this, key phis only by the edge, and keep
                // a per-edge list of phi input `spv::Id`s (with validation for
                // missing entries/duplicates).
                target_phi_idx: u32,
            }

            struct BlockDetails {
                label_id: spv::Id,
                phi_count: u32,
                total_phi_disaggregated_leaf_count: u32,
            }

            // Index IDs declared within the function, first.
            let mut local_id_defs = FxIndexMap::default();
            // `OpPhi`s are also collected here, to assign them per-edge.
            let mut phi_to_values = FxIndexMap::<PhiKey, SmallVec<[spv::Id; 1]>>::default();
            let mut block_details = FxIndexMap::<ControlRegion, BlockDetails>::default();
            let mut has_blocks = false;
            {
                let mut next_param_disaggregated_leaf_idx = 0u32;
                for raw_inst in &raw_insts {
                    let IntraFuncInst {
                        without_ids: spv::Inst { opcode, ref imms },
                        result_id,
                        ref result_type_and_val_cat,
                        ..
                    } = *raw_inst;

                    // FIXME(eddyb) some of the indentation here could be
                    // removed by just relying on `continue` a lot more.
                    if let Some(id) = result_id {
                        // HACK(eddyb) this "error type" allows declaring several
                        // variables below, despite them not being available in
                        // the `OpLabel` case (ideally that case could be handled
                        // early, but it's sandwiched in-between other opcodes).
                        #[derive(Copy, Clone, Debug)]
                        struct OpLabelHasNoResultType;
                        let (result_type, result_type_val_cat) = if opcode == wk.OpLabel {
                            (Err(OpLabelHasNoResultType), Err(OpLabelHasNoResultType))
                        } else {
                            result_type_and_val_cat
                                .as_ref()
                                .map(|(ty, ty_val_cat)| (Ok(*ty), Ok(ty_val_cat)))
                                .ok_or_else(|| {
                                    invalid(
                                        "expected value-producing instruction, \
                                         with a result type",
                                    )
                                })?
                        };

                        let result_disaggregated_leaf_count =
                            result_type_val_cat.map(|ty_val_cat| match ty_val_cat {
                                TypeValueCategory::Direct => 1,
                                TypeValueCategory::Aggregate { leaf_types } => {
                                    u32::try_from(leaf_types.len()).unwrap()
                                }
                            });

                        let alloc_result_disaggregated_leaf_range = |next: &mut u32| {
                            let first = *next;
                            *next = first
                                .checked_add(result_disaggregated_leaf_count.unwrap())
                                .unwrap();
                            first..*next
                        };

                        let mk_result_from_region_inputs_or_data_inst_outputs =
                            |region_or_data_inst: Either<ControlRegion, DataInst>,
                             disaggregated_leaf_range: Range<u32>| {
                                let disaggregated_leaves = ValueRange {
                                    region_or_data_inst,
                                    range: disaggregated_leaf_range,
                                };
                                match result_type_val_cat.unwrap() {
                                    TypeValueCategory::Direct => {
                                        let mut values = disaggregated_leaves.iter();
                                        assert_eq!(values.len(), 1);
                                        LocalIdDef::Value(values.next().unwrap())
                                    }
                                    TypeValueCategory::Aggregate { .. } => {
                                        LocalIdDef::AggregateValue {
                                            ty: result_type.unwrap(),
                                            leaves: disaggregated_leaves,
                                        }
                                    }
                                }
                            };

                        let local_id_def = if opcode == wk.OpFunctionParameter {
                            let param_disaggregated_leaf_range =
                                alloc_result_disaggregated_leaf_range(
                                    &mut next_param_disaggregated_leaf_idx,
                                );

                            let body = match &func_decl.def {
                                // `LocalIdDef`s not needed for declarations.
                                DeclDef::Imported(_) => continue,

                                DeclDef::Present(def) => def.body,
                            };
                            mk_result_from_region_inputs_or_data_inst_outputs(
                                Either::Left(body),
                                param_disaggregated_leaf_range,
                            )
                        } else {
                            let is_entry_block = !has_blocks;
                            has_blocks = true;

                            let func_def_body = match &mut func_decl.def {
                                // Error will be emitted later, below.
                                DeclDef::Imported(_) => continue,
                                DeclDef::Present(def) => def,
                            };

                            if opcode == wk.OpLabel {
                                let block = if is_entry_block {
                                    // A `ControlRegion` was defined earlier,
                                    // to be able to create the `FuncDefBody`.
                                    func_def_body.body
                                } else {
                                    func_def_body.control_regions.define(
                                        &cx,
                                        ControlRegionDef {
                                            inputs: SmallVec::new(),
                                            children: EntityList::empty(),
                                            outputs: SmallVec::new(),
                                        },
                                    )
                                };
                                block_details.insert(
                                    block,
                                    BlockDetails {
                                        label_id: id,
                                        phi_count: 0,
                                        total_phi_disaggregated_leaf_count: 0,
                                    },
                                );
                                LocalIdDef::BlockLabel(block)
                            } else if opcode == wk.OpPhi {
                                let (&current_block, block_details) = match block_details.last_mut()
                                {
                                    Some(entry) => entry,
                                    // Error will be emitted later, below.
                                    None => continue,
                                };

                                let phi_idx = block_details.phi_count;
                                block_details.phi_count = phi_idx.checked_add(1).unwrap();

                                let phi_disaggregated_leaf_range =
                                    alloc_result_disaggregated_leaf_range(
                                        &mut block_details.total_phi_disaggregated_leaf_count,
                                    );

                                assert!(imms.is_empty());
                                // FIXME(eddyb) use `array_chunks` when that's stable.
                                for value_and_source_block_id in raw_inst.ids.chunks(2) {
                                    let &[value_id, source_block_id]: &[_; 2] =
                                        value_and_source_block_id.try_into().unwrap();

                                    phi_to_values
                                        .entry(PhiKey {
                                            source_block_id,
                                            target_block_id: block_details.label_id,
                                            target_phi_idx: phi_idx,
                                        })
                                        .or_default()
                                        .push(value_id);
                                }

                                mk_result_from_region_inputs_or_data_inst_outputs(
                                    Either::Left(current_block),
                                    phi_disaggregated_leaf_range,
                                )
                            } else {
                                // HACK(eddyb) can't get a `DataInst` without
                                // defining it (as a dummy) first.
                                let inst = func_def_body.data_insts.define(
                                    &cx,
                                    DataInstDef {
                                        attrs: AttrSet::default(),
                                        // FIXME(eddyb) cache this form locally.
                                        form: get_op_nop_form(),
                                        inputs: [].into_iter().collect(),
                                    }
                                    .into(),
                                );
                                // FIXME(eddyb) lowering `OpComposite*`, `OpCopyObject` etc.
                                // (anything with aggregate result type) has to be
                                // done here, and not later, because of how it
                                // interacts with ID definitions
                                mk_result_from_region_inputs_or_data_inst_outputs(
                                    Either::Right(inst),
                                    0..result_disaggregated_leaf_count.unwrap(),
                                )
                            }
                        };
                        local_id_defs.insert(id, local_id_def);
                    }
                }
            }

            let mut params = SmallVec::<[_; 8]>::new();

            let mut func_def_body = if has_blocks {
                match &mut func_decl.def {
                    DeclDef::Imported(Import::LinkName(name)) => {
                        return Err(invalid(&format!(
                            "non-empty function %{} decorated as `Import` of {:?}",
                            func_id, &cx[*name]
                        )));
                    }
                    DeclDef::Present(def) => Some(def),
                }
            } else {
                match func_decl.def {
                    DeclDef::Imported(Import::LinkName(_)) => {}
                    DeclDef::Present(_) => {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(&format!(
                            "function %{func_id} lacks any blocks, \
                             but isn't an import either"
                        )));
                    }
                }

                None
            };

            let mut current_block_control_region_and_details = None;
            for (raw_inst_idx, raw_inst) in raw_insts.iter().enumerate() {
                let lookahead_raw_inst = |dist| {
                    raw_inst_idx
                        .checked_add(dist)
                        .and_then(|i| raw_insts.get(i))
                };

                let IntraFuncInst {
                    attrs,
                    ref result_type_and_val_cat,
                    without_ids: spv::Inst { opcode, ref imms },
                    result_id,
                    ref ids,
                } = *raw_inst;

                let invalid = |msg: &str| invalid(&format!("in {}: {}", opcode.name(), msg));

                // FIXME(eddyb) find a more compact name and/or make this a method.
                // FIXME(eddyb) this returns `LocalIdDef` even for global values.
                let lookup_global_or_local_id_for_data_or_control_inst_input =
                    |id| match id_defs.get(&id) {
                        None => {
                            let local_id_def = local_id_defs
                                .get(&id)
                                .cloned()
                                .ok_or_else(|| invalid(&format!("undefined ID %{id}")))?;
                            // HACK(eddyb) change the type of `leaves` within
                            // `LocalIdDef::AggregateValue` to support consts
                            // (see `IdDef::AggregateConst` case just below).
                            Ok(match local_id_def {
                                LocalIdDef::Value(v) => LocalIdDef::Value(v),
                                LocalIdDef::AggregateValue { ty, leaves } => {
                                    LocalIdDef::AggregateValue {
                                        ty,
                                        leaves: Either::Left(leaves.iter()),
                                    }
                                }
                                LocalIdDef::BlockLabel(label) => LocalIdDef::BlockLabel(label),
                            })
                        }
                        Some(&IdDef::Const(ct)) => Ok(LocalIdDef::Value(Value::Const(ct))),
                        Some(IdDef::AggregateConst {
                            whole_const,
                            leaves,
                        }) => {
                            let ty = cx[*whole_const].ty;
                            Ok(LocalIdDef::AggregateValue {
                                ty,
                                leaves: Either::Right(leaves.iter().copied().map(Value::Const)),
                            })
                        }
                        Some(id_def @ IdDef::Type(_)) => Err(invalid(&format!(
                            "unsupported use of {} as an operand for \
                             an instruction in a function",
                            id_def.descr(&cx),
                        ))),
                        Some(id_def @ IdDef::Func(_)) => Err(invalid(&format!(
                            "unsupported use of {} outside `OpFunctionCall`",
                            id_def.descr(&cx),
                        ))),
                        Some(id_def @ IdDef::SpvDebugString(s)) => {
                            if opcode == wk.OpExtInst {
                                // HACK(eddyb) intern `OpString`s as `Const`s on
                                // the fly, as it's a less likely usage than the
                                // `OpLine` one.
                                let ty = cx.intern(TypeDef {
                                    attrs: AttrSet::default(),
                                    ctor: TypeCtor::SpvStringLiteralForExtInst,
                                    ctor_args: [].into_iter().collect(),
                                });
                                let ct = cx.intern(ConstDef {
                                    attrs: AttrSet::default(),
                                    ty,
                                    ctor: ConstCtor::SpvStringLiteralForExtInst(*s),
                                    ctor_args: [].into_iter().collect(),
                                });
                                Ok(LocalIdDef::Value(Value::Const(ct)))
                            } else {
                                Err(invalid(&format!(
                                    "unsupported use of {} outside `OpSource`, \
                                     `OpLine`, or `OpExtInst`",
                                    id_def.descr(&cx),
                                )))
                            }
                        }
                        Some(id_def @ IdDef::SpvExtInstImport(_)) => Err(invalid(&format!(
                            "unsupported use of {} outside `OpExtInst`",
                            id_def.descr(&cx),
                        ))),
                    };

                if opcode == wk.OpFunctionParameter {
                    if current_block_control_region_and_details.is_some() {
                        return Err(invalid(
                            "out of order: `OpFunctionParameter`s should come \
                             before the function's blocks",
                        ));
                    }

                    assert!(imms.is_empty() && ids.is_empty());

                    let (ty, ty_val_cat) = result_type_and_val_cat.as_ref().unwrap();
                    let param_disaggregated_leaves = match ty_val_cat {
                        TypeValueCategory::Direct => {
                            Either::Left([FuncParam { attrs, ty: *ty }].into_iter())
                        }

                        // FIXME(eddyb) this may lose semantic `attrs`.
                        TypeValueCategory::Aggregate { leaf_types } => {
                            Either::Right(leaf_types.iter().map(|&ty| FuncParam {
                                attrs: AttrSet::default(),
                                ty,
                            }))
                        }
                    };

                    params.extend(param_disaggregated_leaves.clone());
                    if let Some(func_def_body) = &mut func_def_body {
                        func_def_body.at_mut_body().def().inputs.extend(
                            param_disaggregated_leaves.map(|FuncParam { attrs, ty }| {
                                ControlRegionInputDecl { attrs, ty }
                            }),
                        );
                    }
                    continue;
                }
                let func_def_body = func_def_body.as_deref_mut().unwrap();

                let is_last_in_block = lookahead_raw_inst(1).map_or(true, |next_raw_inst| {
                    next_raw_inst.without_ids.opcode == wk.OpLabel
                });

                if opcode == wk.OpLabel {
                    if is_last_in_block {
                        return Err(invalid("block lacks terminator instruction"));
                    }

                    // A `ControlRegion` (using an empty `Block` `ControlNode`
                    // as its sole child) was defined earlier,
                    // to be able to have an entry in `local_id_defs`.
                    let control_region = match local_id_defs[&result_id.unwrap()] {
                        LocalIdDef::BlockLabel(control_region) => control_region,
                        _ => unreachable!(),
                    };
                    let current_block_details = &block_details[&control_region];
                    assert_eq!(current_block_details.label_id, result_id.unwrap());
                    current_block_control_region_and_details =
                        Some((control_region, current_block_details));
                    continue;
                }
                let (current_block_control_region, current_block_details) =
                    current_block_control_region_and_details.ok_or_else(|| {
                        invalid("out of order: not expected before the function's blocks")
                    })?;
                let current_block_control_region_def =
                    &mut func_def_body.control_regions[current_block_control_region];

                if is_last_in_block {
                    if opcode.def().category != spec::InstructionCategory::ControlFlow
                        || [wk.OpPhi, wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode)
                    {
                        return Err(invalid(
                            "non-control-flow instruction cannot be used \
                             as the terminator instruction of a block",
                        ));
                    }

                    let mut target_inputs = FxIndexMap::default();
                    let mut record_cfg_edge = |target_block| -> io::Result<()> {
                        use indexmap::map::Entry;

                        let target_block_details = &block_details[&target_block];

                        if target_block_details.phi_count == 0 {
                            return Ok(());
                        }

                        // Only resolve `OpPhi`s exactly once (per target).
                        let target_inputs_entry = match target_inputs.entry(target_block) {
                            Entry::Occupied(_) => return Ok(()),
                            Entry::Vacant(entry) => entry,
                        };

                        let mut target_inputs = SmallVec::with_capacity(
                            target_block_details.total_phi_disaggregated_leaf_count as usize,
                        );
                        for target_phi_idx in 0..target_block_details.phi_count {
                            let phi_key = PhiKey {
                                source_block_id: current_block_details.label_id,
                                target_block_id: target_block_details.label_id,
                                target_phi_idx,
                            };
                            let descr_phi_case = || {
                                format!(
                                    "`OpPhi` (#{} in %{})'s case for source block %{}",
                                    phi_key.target_phi_idx,
                                    phi_key.target_block_id,
                                    phi_key.source_block_id,
                                )
                            };

                            let phi_value_ids = phi_to_values.remove(&phi_key).unwrap_or_default();

                            let phi_value_id = match phi_value_ids[..] {
                                [] => {
                                    return Err(invalid(&format!(
                                        "{} is missing",
                                        descr_phi_case()
                                    )));
                                }
                                [id] => id,
                                [..] => {
                                    return Err(invalid(&format!(
                                        "{} is duplicated",
                                        descr_phi_case()
                                    )));
                                }
                            };

                            match lookup_global_or_local_id_for_data_or_control_inst_input(
                                phi_value_id,
                            )? {
                                LocalIdDef::Value(v) => target_inputs.push(v),
                                LocalIdDef::AggregateValue { leaves, .. } => {
                                    target_inputs.extend(leaves);
                                }
                                LocalIdDef::BlockLabel { .. } => {
                                    return Err(invalid(&format!(
                                        "unsupported use of block label as the value for {}",
                                        descr_phi_case()
                                    )));
                                }
                            }
                        }

                        target_inputs_entry.insert(target_inputs);

                        Ok(())
                    };

                    // Split the operands into value inputs (e.g. a branch's
                    // condition or an `OpSwitch`'s selector) and target blocks.
                    let mut inputs = SmallVec::new();
                    let mut targets = SmallVec::new();
                    for &id in ids {
                        match lookup_global_or_local_id_for_data_or_control_inst_input(id)? {
                            LocalIdDef::Value(v) => {
                                if !targets.is_empty() {
                                    return Err(invalid(
                                        "out of order: value operand \
                                         after target label ID",
                                    ));
                                }
                                inputs.push(v);
                            }
                            LocalIdDef::AggregateValue { leaves, .. } => {
                                // Returns are "lossily" disaggregated, just like
                                // function's signatures and calls to them.
                                if opcode == wk.OpReturnValue {
                                    inputs.extend(leaves);
                                } else {
                                    return Err(invalid(
                                        "unsupported aggregate value operand, \
                                         in non-return terminator instruction",
                                    ));
                                }
                            }
                            LocalIdDef::BlockLabel(target) => {
                                record_cfg_edge(target)?;
                                targets.push(target);
                            }
                        }
                    }

                    let kind = if opcode == wk.OpUnreachable {
                        assert!(targets.is_empty() && inputs.is_empty());
                        cfg::ControlInstKind::Unreachable
                    } else if [wk.OpReturn, wk.OpReturnValue].contains(&opcode) {
                        assert!(targets.is_empty());
                        cfg::ControlInstKind::Return
                    } else if targets.is_empty() {
                        cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(
                            raw_inst.without_ids.clone(),
                        ))
                    } else if opcode == wk.OpBranch {
                        assert_eq!((targets.len(), inputs.len()), (1, 0));
                        cfg::ControlInstKind::Branch
                    } else if opcode == wk.OpBranchConditional {
                        assert_eq!((targets.len(), inputs.len()), (2, 1));
                        cfg::ControlInstKind::SelectBranch(SelectionKind::BoolCond)
                    } else if opcode == wk.OpSwitch {
                        cfg::ControlInstKind::SelectBranch(SelectionKind::SpvInst(
                            raw_inst.without_ids.clone(),
                        ))
                    } else {
                        return Err(invalid("unsupported control-flow instruction"));
                    };

                    func_def_body
                        .unstructured_cfg
                        .as_mut()
                        .unwrap()
                        .control_inst_on_exit_from
                        .insert(
                            current_block_control_region,
                            cfg::ControlInst {
                                attrs,
                                kind,
                                inputs,
                                targets,
                                target_inputs,
                            },
                        );
                } else if opcode == wk.OpPhi {
                    if !current_block_control_region_def.children.is_empty() {
                        return Err(invalid(
                            "out of order: `OpPhi`s should come before \
                             the rest of the block's instructions",
                        ));
                    }

                    let (ty, ty_val_cat) = result_type_and_val_cat.as_ref().unwrap();
                    match ty_val_cat {
                        TypeValueCategory::Direct => {
                            current_block_control_region_def
                                .inputs
                                .push(ControlRegionInputDecl { attrs, ty: *ty });
                        }

                        // FIXME(eddyb) this may lose semantic `attrs`.
                        TypeValueCategory::Aggregate { leaf_types } => {
                            current_block_control_region_def
                                .inputs
                                .extend(leaf_types.iter().map(|&ty| ControlRegionInputDecl {
                                    attrs: AttrSet::default(),
                                    ty,
                                }));
                        }
                    }
                } else if [wk.OpSelectionMerge, wk.OpLoopMerge].contains(&opcode) {
                    let is_second_to_last_in_block = lookahead_raw_inst(2)
                        .map_or(true, |next_raw_inst| {
                            next_raw_inst.without_ids.opcode == wk.OpLabel
                        });

                    if !is_second_to_last_in_block {
                        return Err(invalid(
                            "out of order: a merge instruction should be the last \
                             instruction before the block's terminator",
                        ));
                    }

                    // HACK(eddyb) merges are ignored - this may be lossy,
                    // especially wrt the `SelectionControl` and `LoopControl`
                    // operands, but it's not obvious how they should map to
                    // some "structured regions" replacement for the CFG.
                } else {
                    let mut ids = &ids[..];
                    let mut kind = if opcode == wk.OpFunctionCall {
                        assert!(imms.is_empty());
                        let callee_id = ids[0];
                        let maybe_callee = id_defs
                            .get(&callee_id)
                            .map(|id_def| match *id_def {
                                IdDef::Func(func) => Ok(func),
                                _ => Err(id_def.descr(&cx)),
                            })
                            .transpose()
                            .map_err(|descr| {
                                invalid(&format!(
                                    "unsupported use of {descr} as the `OpFunctionCall` callee"
                                ))
                            })?;

                        match maybe_callee {
                            Some(callee) => {
                                ids = &ids[1..];
                                DataInstKind::FuncCall(callee)
                            }

                            // HACK(eddyb) this should be an error, but it shows
                            // up in Rust-GPU output (likely a zombie?).
                            None => DataInstKind::SpvInst(
                                raw_inst.without_ids.clone(),
                                spv::InstLowering::default(),
                            ),
                        }
                    } else if opcode == wk.OpExtInst {
                        let ext_set_id = ids[0];
                        ids = &ids[1..];

                        let inst = match imms[..] {
                            [spv::Imm::Short(kind, inst)] => {
                                assert_eq!(kind, wk.LiteralExtInstInteger);
                                inst
                            }
                            _ => unreachable!(),
                        };

                        let ext_set = match id_defs.get(&ext_set_id) {
                            Some(&IdDef::SpvExtInstImport(name)) => Ok(name),
                            Some(id_def) => Err(id_def.descr(&cx)),
                            None => Err(format!("unknown ID %{ext_set_id}")),
                        }
                        .map_err(|descr| {
                            invalid(&format!(
                                "unsupported use of {descr} as the `OpExtInst` \
                                 extended instruction set ID"
                            ))
                        })?;

                        DataInstKind::SpvExtInst {
                            ext_set,
                            inst,
                            lowering: spv::InstLowering::default(),
                        }
                    } else {
                        DataInstKind::SpvInst(
                            raw_inst.without_ids.clone(),
                            spv::InstLowering::default(),
                        )
                    };

                    let mut spv_inst_lowering = match &mut kind {
                        DataInstKind::SpvInst(_, lowering)
                        | DataInstKind::SpvExtInst { lowering, .. } => Some(lowering),

                        // NOTE(eddyb) function signatures and calls keep their
                        // disaggregation even when lifting back to SPIR-V, so
                        // no `spv::InstLowering` is tracked for them.
                        DataInstKind::FuncCall(_) => None,

                        DataInstKind::QPtr(_) => unreachable!(),
                    };

                    let output_types = result_id
                        .map(|_| {
                            let (ty, ty_val_cat) = result_type_and_val_cat.as_ref().unwrap();
                            match ty_val_cat {
                                TypeValueCategory::Direct => [*ty].into_iter().collect(),
                                TypeValueCategory::Aggregate { leaf_types } => {
                                    if let Some(lowering) = &mut spv_inst_lowering {
                                        lowering.disaggregated_output = Some(*ty);
                                    }
                                    leaf_types.iter().copied().collect()
                                }
                            }
                        })
                        .unwrap_or_default();

                    let mut inputs = SmallVec::with_capacity(ids.len());
                    for &id in ids {
                        match lookup_global_or_local_id_for_data_or_control_inst_input(id)? {
                            LocalIdDef::Value(v) => inputs.push(v),
                            LocalIdDef::AggregateValue { ty, leaves } => {
                                let start = inputs.len() as u32;
                                inputs.extend(leaves);
                                let end = inputs.len() as u32;

                                if let Some(lowering) = &mut spv_inst_lowering {
                                    lowering.disaggregated_inputs.push((start..end, ty));
                                }
                            }
                            LocalIdDef::BlockLabel { .. } => {
                                return Err(invalid(
                                    "unsupported use of block label as a value, \
                                     in non-terminator instruction",
                                ));
                            }
                        }
                    }

                    let data_inst_def = DataInstDef {
                        attrs,
                        form: cx.intern(DataInstFormDef { kind, output_types }),
                        inputs,
                    };
                    let inst = match result_id {
                        Some(id) => match local_id_defs[&id] {
                            LocalIdDef::Value(Value::DataInstOutput { inst, .. })
                            | LocalIdDef::AggregateValue {
                                leaves:
                                    ValueRange {
                                        region_or_data_inst: Either::Right(inst),
                                        ..
                                    },
                                ..
                            } => {
                                // A dummy was defined earlier, to be able to
                                // have an entry in `local_id_defs`.
                                func_def_body.data_insts[inst] = data_inst_def.into();

                                inst
                            }
                            _ => unreachable!(),
                        },
                        None => func_def_body.data_insts.define(&cx, data_inst_def.into()),
                    };

                    let current_block_control_node = current_block_control_region_def
                        .children
                        .iter()
                        .last
                        .filter(|&last_node| {
                            matches!(
                                func_def_body.control_nodes[last_node].kind,
                                ControlNodeKind::Block { .. }
                            )
                        })
                        .unwrap_or_else(|| {
                            let block_node = func_def_body.control_nodes.define(
                                &cx,
                                ControlNodeDef {
                                    kind: ControlNodeKind::Block {
                                        insts: EntityList::empty(),
                                    },
                                    outputs: SmallVec::new(),
                                }
                                .into(),
                            );
                            current_block_control_region_def
                                .children
                                .insert_last(block_node, &mut func_def_body.control_nodes);
                            block_node
                        });
                    match &mut func_def_body.control_nodes[current_block_control_node].kind {
                        ControlNodeKind::Block { insts } => {
                            insts.insert_last(inst, &mut func_def_body.data_insts);
                        }
                        _ => unreachable!(),
                    }
                }
            }

            // FIXME(eddyb) all functions should have the appropriate number of
            // `OpFunctionParameter`, even imports.
            if !params.is_empty() {
                if func_decl.params.len() != params.len() {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{}, param count differs between `OpTypeFunction` ({}) \
                         and `OpFunctionParameter`s ({})",
                        func_id,
                        func_decl.params.len(),
                        params.len(),
                    )));
                }

                for (i, (func_decl_param, param)) in
                    func_decl.params.iter_mut().zip(params).enumerate()
                {
                    func_decl_param.attrs = param.attrs;
                    if func_decl_param.ty != param.ty {
                        // FIXME(remove) embed IDs in errors by moving them to the
                        // `let invalid = |...| ...;` closure that wraps insts.
                        return Err(invalid(&format!(
                            "in %{}, param {}'s type differs between \
                             `OpTypeFunction` (expected) and \
                             `OpFunctionParameter` (found):\n\n{}",
                            func_id,
                            i,
                            print::Plan::for_root(
                                &cx,
                                &print::ExpectedVsFound {
                                    expected: func_decl_param.ty,
                                    found: param.ty,
                                }
                            )
                            .pretty_print()
                        )));
                    }
                }
            }

            if !phi_to_values.is_empty() {
                let mut edges = phi_to_values
                    .keys()
                    .map(|key| format!("%{} -> %{}", key.source_block_id, key.target_block_id))
                    .collect::<Vec<_>>();
                edges.dedup();
                // FIXME(remove) embed IDs in errors by moving them to the
                // `let invalid = |...| ...;` closure that wraps insts.
                return Err(invalid(&format!(
                    "in %{}, `OpPhi`s refer to non-existent edges: {}",
                    func_id,
                    edges.join(", ")
                )));
            }

            // Sanity-check the entry block.
            if let Some(func_def_body) = func_def_body {
                if block_details[&func_def_body.body].phi_count > 0 {
                    // FIXME(remove) embed IDs in errors by moving them to the
                    // `let invalid = |...| ...;` closure that wraps insts.
                    return Err(invalid(&format!(
                        "in %{func_id}, the entry block contains `OpPhi`s"
                    )));
                }
            }
        }

        assert!(module.exports.is_empty());
        module.exports = pending_exports
            .into_iter()
            .map(|export| match export {
                Export::Linkage { name, target_id } => {
                    let exportee = match id_defs.get(&target_id) {
                        Some(id_def @ &IdDef::Const(ct)) => match cx[ct].ctor {
                            ConstCtor::PtrToGlobalVar(gv) => Ok(Exportee::GlobalVar(gv)),
                            _ => Err(id_def.descr(&cx)),
                        },
                        Some(&IdDef::Func(func)) => Ok(Exportee::Func(func)),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{target_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `LinkageAttributes` target"
                        ))
                    })?;

                    Ok((ExportKey::LinkName(name), exportee))
                }

                Export::EntryPoint {
                    func_id,
                    imms,
                    interface_ids,
                } => {
                    let func = match id_defs.get(&func_id) {
                        Some(&IdDef::Func(func)) => Ok(func),
                        Some(id_def) => Err(id_def.descr(&cx)),
                        None => Err(format!("unknown ID %{func_id}")),
                    }
                    .map_err(|descr| {
                        invalid(&format!(
                            "unsupported use of {descr} as the `OpEntryPoint` target"
                        ))
                    })?;
                    let interface_global_vars = interface_ids
                        .into_iter()
                        .map(|id| match id_defs.get(&id) {
                            Some(id_def @ &IdDef::Const(ct)) => match cx[ct].ctor {
                                ConstCtor::PtrToGlobalVar(gv) => Ok(gv),
                                _ => Err(id_def.descr(&cx)),
                            },
                            Some(id_def) => Err(id_def.descr(&cx)),
                            None => Err(format!("unknown ID %{id}")),
                        })
                        .map(|result| {
                            result.map_err(|descr| {
                                invalid(&format!(
                                    "unsupported use of {descr} as an `OpEntryPoint` interface variable"
                                ))
                            })
                        })
                        .collect::<Result<_, _>>()?;
                    Ok((
                        ExportKey::SpvEntryPoint {
                            imms,
                            interface_global_vars,
                        },
                        Exportee::Func(func),
                    ))
                }
            })
            .collect::<io::Result<_>>()?;

        Ok(module)
    }
}
