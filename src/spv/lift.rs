//! SPIR-T to SPIR-V lifting.

use crate::func_at::FuncAt;
use crate::spv::{self, spec};
use crate::visit::{InnerVisit, Visitor};
use crate::{
    cfg, AddrSpace, Attr, AttrSet, Const, ConstCtor, ConstDef, Context, ControlNode,
    ControlNodeKind, ControlNodeOutputDecl, ControlRegion, ControlRegionInputDecl, DataInst,
    DataInstDef, DataInstForm, DataInstFormDef, DataInstKind, DeclDef, EntityList,
    EntityOrientedDenseMap, ExportKey, Exportee, Func, FuncDecl, FuncDefBody, FuncParam,
    FxIndexMap, FxIndexSet, GlobalVar, GlobalVarDefBody, Import, Module, ModuleDebugInfo,
    ModuleDialect, SelectionKind, Type, TypeCtor, TypeCtorArg, TypeDef, Value,
};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;
use std::rc::Rc;
use std::{io, iter, mem, slice};

impl spv::Dialect {
    fn capability_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.capabilities.iter().map(move |&cap| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpCapability,
                imms: iter::once(spv::Imm::Short(wk.Capability, cap)).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }

    pub fn extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.extensions.iter().map(move |ext| spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpExtension,
                imms: spv::encode_literal_string(ext).collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })
    }
}

impl spv::ModuleDebugInfo {
    fn source_extension_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.source_extensions
            .iter()
            .map(move |ext| spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpSourceExtension,
                    imms: spv::encode_literal_string(ext).collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [].into_iter().collect(),
            })
    }

    fn module_processed_insts(&self) -> impl Iterator<Item = spv::InstWithIds> + '_ {
        let wk = &spec::Spec::get().well_known;
        self.module_processes
            .iter()
            .map(move |proc| spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpModuleProcessed,
                    imms: spv::encode_literal_string(proc).collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [].into_iter().collect(),
            })
    }
}

struct IdAllocator<'a, AI: FnMut() -> spv::Id> {
    cx: &'a Context,
    module: &'a Module,

    /// ID allocation callback, kept as a closure (instead of having its state
    /// be part of `IdAllocator`) to avoid misuse.
    alloc_id: AI,

    ids: ModuleIds<'a>,

    data_inst_forms_seen: FxIndexSet<DataInstForm>,
    global_vars_seen: FxIndexSet<GlobalVar>,

    cached_spv_aggregates: FxHashMap<Type, SpvAggregate>,
}

#[derive(Default)]
struct ModuleIds<'a> {
    ext_inst_imports: BTreeMap<&'a str, spv::Id>,
    debug_strings: BTreeMap<&'a str, spv::Id>,

    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    globals: FxIndexMap<Global, spv::Id>,
    // FIXME(eddyb) use `EntityOrientedDenseMap` here.
    funcs: FxIndexMap<Func, FuncIds<'a>>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum Global {
    Type(Type),
    Const(Const),
}

// FIXME(eddyb) should this use ID ranges instead of `SmallVec<[spv::Id; 4]>`?
// FIXME(eddyb) this is inconsistently named with `FuncBodyLifting`.
struct FuncIds<'a> {
    spv_func_ret_type: Type,
    // FIXME(eddyb) should we even be interning an `OpTypeFunction` in `Context`?
    // (it's easier this way, but it could also be tracked in `ModuleIds`)
    spv_func_type: Type,

    func_id: spv::Id,
    param_ids: SmallVec<[spv::Id; 4]>,

    body: Option<FuncBodyLifting<'a>>,
}

impl<AI: FnMut() -> spv::Id> Visitor<'_> for IdAllocator<'_, AI> {
    fn visit_attr_set_use(&mut self, attrs: AttrSet) {
        self.visit_attr_set_def(&self.cx[attrs]);
    }
    fn visit_type_use(&mut self, ty: Type) {
        let global = Global::Type(ty);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ty_def = &self.cx[ty];
        match ty_def.ctor {
            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            TypeCtor::QPtr => {
                unreachable!("`TypeCtor::QPtr` should be legalized away before lifting");
            }

            TypeCtor::SpvInst(_) => {}
            TypeCtor::SpvStringLiteralForExtInst => {
                unreachable!(
                    "`TypeCtor::SpvStringLiteralForExtInst` should not be used \
                     as a type outside of `ConstCtor::SpvStringLiteralForExtInst`"
                );
            }
        }
        self.visit_type_def(ty_def);
        self.ids.globals.insert(global, (self.alloc_id)());
    }
    fn visit_const_use(&mut self, ct: Const) {
        let global = Global::Const(ct);
        if self.ids.globals.contains_key(&global) {
            return;
        }
        let ct_def = &self.cx[ct];
        match ct_def.ctor {
            ConstCtor::PtrToGlobalVar(_) | ConstCtor::SpvInst(_) => {
                self.visit_const_def(ct_def);
                self.ids.globals.insert(global, (self.alloc_id)());
            }

            // HACK(eddyb) because this is an `OpString` and needs to go earlier
            // in the module than any `OpConstant*`, it needs to be special-cased,
            // without visiting its type, or an entry in `self.globals`.
            ConstCtor::SpvStringLiteralForExtInst(s) => {
                let ConstDef {
                    attrs,
                    ty,
                    ctor: _,
                    ctor_args,
                } = ct_def;

                assert!(*attrs == AttrSet::default());
                assert!(
                    self.cx[*ty]
                        == TypeDef {
                            attrs: AttrSet::default(),
                            ctor: TypeCtor::SpvStringLiteralForExtInst,
                            ctor_args: SmallVec::new(),
                        }
                );
                assert!(ctor_args.is_empty());

                self.ids
                    .debug_strings
                    .entry(&self.cx[s])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
    }
    fn visit_data_inst_form_use(&mut self, data_inst_form: DataInstForm) {
        if self.data_inst_forms_seen.insert(data_inst_form) {
            self.visit_data_inst_form_def(&self.cx[data_inst_form]);
        }
    }

    fn visit_global_var_use(&mut self, gv: GlobalVar) {
        if self.global_vars_seen.insert(gv) {
            self.visit_global_var_decl(&self.module.global_vars[gv]);
        }
    }
    fn visit_func_use(&mut self, func: Func) {
        if self.ids.funcs.contains_key(&func) {
            return;
        }
        let func_decl = &self.module.funcs[func];

        // Synthesize an `OpTypeFunction` type (that SPIR-T itself doesn't carry).
        let wk = &spec::Spec::get().well_known;
        let spv_func_ret_type = match &func_decl.ret_types[..] {
            &[ty] => ty,
            // Reaggregate multiple return types into an `OpTypeStruct`.
            ret_types => {
                let opcode = if ret_types.is_empty() {
                    wk.OpTypeVoid
                } else {
                    wk.OpTypeStruct
                };
                self.cx.intern(TypeDef {
                    attrs: AttrSet::default(),
                    ctor: TypeCtor::SpvInst(opcode.into()),
                    ctor_args: ret_types.iter().copied().map(TypeCtorArg::Type).collect(),
                })
            }
        };
        let spv_func_type = self.cx.intern(TypeDef {
            attrs: AttrSet::default(),
            ctor: TypeCtor::SpvInst(wk.OpTypeFunction.into()),
            ctor_args: iter::once(spv_func_ret_type)
                .chain(func_decl.params.iter().map(|param| param.ty))
                .map(TypeCtorArg::Type)
                .collect(),
        });
        self.visit_type_use(spv_func_type);

        // NOTE(eddyb) inserting first produces a different function ordering
        // overall in the final module, but the order doesn't matter, and we
        // need to avoid infinite recursion for recursive functions.
        self.ids.funcs.insert(
            func,
            FuncIds {
                spv_func_ret_type,
                spv_func_type,
                func_id: (self.alloc_id)(),
                param_ids: func_decl.params.iter().map(|_| (self.alloc_id)()).collect(),
                body: None,
            },
        );

        self.visit_func_decl(func_decl);

        // Handle the body last, to minimize recursion hazards (see comment above),
        // and to allow `FuncBodyLifting` to look up its dependencies in `self.ids`.
        match &func_decl.def {
            DeclDef::Imported(_) => {}
            DeclDef::Present(func_def_body) => {
                let func_body_lifting = FuncBodyLifting::from_func_def_body(self, func_def_body);
                self.ids.funcs.get_mut(&func).unwrap().body = Some(func_body_lifting);
            }
        }
    }

    fn visit_spv_module_debug_info(&mut self, debug_info: &spv::ModuleDebugInfo) {
        for sources in debug_info.source_languages.values() {
            // The file operand of `OpSource` has to point to an `OpString`.
            for &s in sources.file_contents.keys() {
                self.ids
                    .debug_strings
                    .entry(&self.cx[s])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
    }
    fn visit_attr(&mut self, attr: &Attr) {
        match *attr {
            Attr::Diagnostics(_)
            | Attr::QPtr(_)
            | Attr::SpvAnnotation { .. }
            | Attr::SpvBitflagsOperand(_) => {}
            Attr::SpvDebugLine { file_path, .. } => {
                self.ids
                    .debug_strings
                    .entry(&self.cx[file_path.0])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
        attr.inner_visit_with(self);
    }

    fn visit_data_inst_form_def(&mut self, data_inst_form_def: &DataInstFormDef) {
        #[allow(clippy::match_same_arms)]
        match data_inst_form_def.kind {
            // FIXME(eddyb) this should be a proper `Result`-based error instead,
            // and/or `spv::lift` should mutate the module for legalization.
            DataInstKind::QPtr(_) => {
                unreachable!("`DataInstKind::QPtr` should be legalized away before lifting");
            }

            DataInstKind::FuncCall(_) => {}

            DataInstKind::SpvInst(..) => {}
            DataInstKind::SpvExtInst { ext_set, .. } => {
                self.ids
                    .ext_inst_imports
                    .entry(&self.cx[ext_set])
                    .or_insert_with(&mut self.alloc_id);
            }
        }
        data_inst_form_def.inner_visit_with(self);
    }
}

/// Information necessary for regenerating values of some SPIR-V "aggregate" type
/// (`OpTypeStruct`/`OpTypeArray`) from the disaggregated form in SPIR-T, and
/// also any extractions implied by [`Value::DataInstOutput`] uses of the leaves.
#[derive(Clone)]
struct SpvAggregate {
    leaves: Rc<SmallVec<[SpvAggregateLeaf; 2]>>,
}

struct SpvAggregateLeaf {
    /// Path through the whole parent [`SpvAggregate`], to this leaf, consisting
    /// of `OpTypeStruct` field indices and/or `OpTypeArray` element indices.
    //
    // NOTE(eddyb) it may seem possible to have `O(1)` leaves that indicate
    // whether they start/end nested aggregates, but each level can have some
    // number of ZSTs (contributing zero leaves), so this compromise is needed.
    path: SmallVec<[u32; 4]>,
}

impl SpvAggregateLeaf {
    fn op_composite_insert(&self) -> spv::Inst {
        let wk = &spec::Spec::get().well_known;
        let int_imm = |i| spv::Imm::Short(wk.LiteralInteger, i);
        spv::Inst {
            opcode: wk.OpCompositeInsert,
            imms: self.path.iter().copied().map(int_imm).collect(),
        }
    }
    fn op_composite_extract(&self) -> spv::Inst {
        let wk = &spec::Spec::get().well_known;
        let int_imm = |i| spv::Imm::Short(wk.LiteralInteger, i);
        spv::Inst {
            opcode: wk.OpCompositeExtract,
            imms: self.path.iter().copied().map(int_imm).collect(),
        }
    }
}

impl<AI: FnMut() -> spv::Id> IdAllocator<'_, AI> {
    fn spv_aggregate(&mut self, ty: Type) -> SpvAggregate {
        if let Some(cached) = self.cached_spv_aggregates.get(&ty).cloned() {
            return cached;
        }
        let spv_aggregate = self.unchached_try_spv_aggregate(ty).unwrap_or_else(|msg| {
            unreachable!(
                "spv::lift::IdAllocator::spv_aggregate: {msg}: {}",
                // HACK(eddyb) the `DiagMsgPart` is only used because `Type`
                // doesn't fit the pretty-printer API wrt the `AttrsAndDef` type.
                crate::print::Plan::for_root(self.cx, &vec![crate::DiagMsgPart::Type(ty)])
                    .pretty_print()
            );
        });
        self.cached_spv_aggregates.insert(ty, spv_aggregate.clone());
        spv_aggregate
    }

    // HACK(eddyb) this allows detecting leaves (by turning `Err` into `None`).
    fn maybe_spv_aggregate(&mut self, ty: Type) -> Option<SpvAggregate> {
        if let Some(cached) = self.cached_spv_aggregates.get(&ty).cloned() {
            return Some(cached);
        }
        let spv_aggregate = self.unchached_try_spv_aggregate(ty).ok()?;
        self.cached_spv_aggregates.insert(ty, spv_aggregate.clone());
        Some(spv_aggregate)
    }

    fn unchached_try_spv_aggregate(&mut self, ty: Type) -> Result<SpvAggregate, &'static str> {
        let wk = &spec::Spec::get().well_known;

        let ty_def = &self.cx[ty];

        let mut leaves = SmallVec::new();
        match &ty_def.ctor {
            TypeCtor::SpvInst(spv_inst) if spv_inst.opcode == wk.OpTypeStruct => {
                for (i, &arg) in ty_def.ctor_args.iter().enumerate() {
                    let field_type = match arg {
                        TypeCtorArg::Type(ty) => ty,
                        TypeCtorArg::Const(_) => {
                            return Err("`OpTypeStruct` with invalid (non-type) operands");
                        }
                    };
                    let field_path_prefix = [u32::try_from(i).unwrap()];
                    match self.maybe_spv_aggregate(field_type) {
                        Some(field_aggregate) => {
                            leaves.extend(field_aggregate.leaves.iter().map(|field_leaf| {
                                SpvAggregateLeaf {
                                    path: field_path_prefix
                                        .into_iter()
                                        .chain(field_leaf.path.iter().copied())
                                        .collect(),
                                }
                            }));
                        }
                        None => leaves.push(SpvAggregateLeaf {
                            path: field_path_prefix.into_iter().collect(),
                        }),
                    }
                }
            }
            TypeCtor::SpvInst(spv_inst) if spv_inst.opcode == wk.OpTypeArray => {
                let (elem_type, len) = match ty_def.ctor_args[..] {
                    [TypeCtorArg::Type(elem_type), TypeCtorArg::Const(len)] => (elem_type, len),
                    _ => return Err("`OpTypeArray` with invalid operands"),
                };

                let elem_aggregate = self.maybe_spv_aggregate(elem_type);

                let fixed_len = match &self.cx[len].ctor {
                    ConstCtor::SpvInst(spv_inst)
                        if spv_inst.opcode == wk.OpConstant && spv_inst.imms.len() == 1 =>
                    {
                        match spv_inst.imms[..] {
                            [spv::Imm::Short(_, x)] => x,
                            _ => unreachable!(),
                        }
                    }
                    _ => return Err("`OpTypeArray` with specialization constant length"),
                };

                let leaf_count = elem_aggregate
                    .as_ref()
                    .map_or(1, |elem_aggregate| elem_aggregate.leaves.len());
                if let Some(capacity) = leaf_count.checked_mul(fixed_len as usize) {
                    leaves.reserve(capacity);
                }

                for i in 0..fixed_len {
                    let elem_path_prefix = [i];
                    match &elem_aggregate {
                        Some(elem_aggregate) => {
                            leaves.extend(elem_aggregate.leaves.iter().map(|elem_leaf| {
                                SpvAggregateLeaf {
                                    path: elem_path_prefix
                                        .into_iter()
                                        .chain(elem_leaf.path.iter().copied())
                                        .collect(),
                                }
                            }));
                        }
                        None => leaves.push(SpvAggregateLeaf {
                            path: elem_path_prefix.into_iter().collect(),
                        }),
                    }
                }
            }
            _ => return Err("type other than `OpTypeStruct`/`OpTypeArray`"),
        }

        Ok(SpvAggregate {
            leaves: Rc::new(leaves),
        })
    }
}

// FIXME(eddyb) this is inconsistently named with `FuncIds`.
struct FuncBodyLifting<'a> {
    region_inputs_source: EntityOrientedDenseMap<ControlRegion, RegionInputsSource>,
    data_insts: EntityOrientedDenseMap<DataInst, DataInstLifting>,

    label_ids: FxHashMap<CfgPoint, spv::Id>,
    blocks: FxIndexMap<CfgPoint, BlockLifting<'a>>,
}

/// What determines the values for [`Value::ControlRegionInput`]s, for a specific
/// region (effectively the subset of "region parents" that support inputs).
///
/// Note that this is not used when a [`cfg::ControlInst`] has `target_inputs`,
/// and the target [`ControlRegion`] itself has phis for its `inputs`.
enum RegionInputsSource {
    FuncParams,
    LoopHeaderPhis(ControlNode),
}

struct DataInstLifting {
    result_id: Option<spv::Id>,

    /// If the SPIR-V result type is "aggregate" (`OpTypeStruct`/`OpTypeArray`),
    /// this describes how to extract its leaves, which is necessary as on the
    /// SPIR-T side, [`Value::DataInstOutput`] can only refer to individual leaves.
    disaggregate_result: Option<DisaggregateToLeaves>,

    /// `reaggregate_inputs[i]` describes how to recreate the "aggregate" value
    /// demanded by [`spv::InstLowering`]'s `disaggregated_inputs[i]`.
    reaggregate_inputs: SmallVec<[ReaggregateFromLeaves; 1]>,
}

/// All the information necessary to decompose a SPIR-V "aggregate" value into
/// its leaves, with one `OpCompositeExtract` per leaf.
//
// FIXME(eddyb) it might be more efficient to only extract actually used leaves,
// or chain partial extracts following nesting structure - but this is simpler.
struct DisaggregateToLeaves {
    aggregate: SpvAggregate,
    // FIXME(eddyb) should this use an ID range instead of `SmallVec<[spv::Id; 4]>`?
    op_composite_extract_result_ids: SmallVec<[spv::Id; 4]>,
}

/// All the information necessary to recreate a SPIR-V "aggregate" value, with
/// one `OpCompositeInsert` per leaf (starting with an `OpUndef` of that type).
//
// FIXME(eddyb) it might be more efficient to use other strategies, such as
// `OpCompositeConstruct`, special-casing constants, reusing whole results
// of other `DataInstDef`s with an aggregate result, etc. - but this is simpler
// for now, and it reuses the "one instruction per leaf" used for extractions.
struct ReaggregateFromLeaves {
    aggregate: SpvAggregate,
    op_undef: Const,
    // FIXME(eddyb) should this use an ID range instead of `SmallVec<[spv::Id; 4]>`?
    op_composite_insert_result_ids: SmallVec<[spv::Id; 4]>,
}

/// Any of the possible points in structured or unstructured SPIR-T control-flow,
/// that may require a separate SPIR-V basic block.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
enum CfgPoint {
    RegionEntry(ControlRegion),
    RegionExit(ControlRegion),

    ControlNodeEntry(ControlNode),
    ControlNodeExit(ControlNode),
}

struct BlockLifting<'a> {
    phis: SmallVec<[Phi; 2]>,
    insts: SmallVec<[EntityList<DataInst>; 1]>,
    terminator: Terminator<'a>,
}

struct Phi {
    attrs: AttrSet,
    ty: Type,

    result_id: spv::Id,
    cases: FxIndexMap<CfgPoint, Value>,

    // HACK(eddyb) used for `Loop` `initial_inputs`, to indicate that any edge
    // to the `Loop` (other than the backedge, which is already in `cases`)
    // should automatically get an entry into `cases`, with this value.
    default_value: Option<Value>,
}

/// Similar to [`cfg::ControlInst`], except:
/// * `targets` use [`CfgPoint`]s instead of [`ControlRegion`]s, to be able to
///   reach any of the SPIR-V blocks being created during lifting
/// * φ ("phi") values can be provided for targets regardless of "which side" of
///   the structured control-flow they are for ("region input" vs "node output")
/// * optional `merge` (for `OpSelectionMerge`/`OpLoopMerge`)
/// * existing data is borrowed (from the [`FuncDefBody`](crate::FuncDefBody)),
///   wherever possible
struct Terminator<'a> {
    attrs: AttrSet,

    kind: Cow<'a, cfg::ControlInstKind>,

    /// If this is a [`cfg::ControlInstKind::Return`] with `inputs.len() > 1`,
    /// this ID is for the `OpCompositeConstruct` needed to produce the single
    /// `OpTypStruct` (`spv_func_ret_type`) value required by `OpReturnValue`.
    reaggregated_return_value_id: Option<spv::Id>,

    // FIXME(eddyb) use `Cow` or something, but ideally the "owned" case always
    // has at most one input, so allocating a whole `Vec` for that seems unwise.
    inputs: SmallVec<[Value; 2]>,

    // FIXME(eddyb) change the inline size of this to fit most instructions.
    targets: SmallVec<[CfgPoint; 4]>,

    target_phi_values: FxIndexMap<CfgPoint, &'a [Value]>,

    merge: Option<Merge<CfgPoint>>,
}

#[derive(Copy, Clone)]
enum Merge<L> {
    Selection(L),

    Loop {
        /// The label just after the whole loop, i.e. the `break` target.
        loop_merge: L,

        /// A label that the back-edge block post-dominates, i.e. some point in
        /// the loop body where looping around is inevitable (modulo `break`ing
        /// out of the loop through a `do`-`while`-style conditional back-edge).
        ///
        /// SPIR-V calls this "the `continue` target", but unlike other aspects
        /// of SPIR-V "structured control-flow", there can be multiple valid
        /// choices (any that fit the post-dominator/"inevitability" definition).
        //
        // FIXME(eddyb) https://github.com/EmbarkStudios/spirt/pull/10 tried to
        // set this to the loop body entry, but that may not be valid if the loop
        // body actually diverges, because then the loop body exit will still be
        // post-dominating the back-edge *but* the loop body itself wouldn't have
        // any relationship between its entry and its *unreachable* exit.
        loop_continue: L,
    },
}

/// Helper type for deep traversal of the CFG (as a graph of [`CfgPoint`]s), which
/// tracks the necessary context for navigating a [`ControlRegion`]/[`ControlNode`].
#[derive(Copy, Clone)]
struct CfgCursor<'p, P = CfgPoint> {
    point: P,
    parent: Option<&'p CfgCursor<'p, ControlParent>>,
}

enum ControlParent {
    Region(ControlRegion),
    ControlNode(ControlNode),
}

impl<'a, 'p> FuncAt<'a, CfgCursor<'p>> {
    /// Return the next [`CfgPoint`] (wrapped in [`CfgCursor`]) in a linear
    /// chain within structured control-flow (i.e. no branching to child regions).
    fn unique_successor(self) -> Option<CfgCursor<'p>> {
        let cursor = self.position;
        match cursor.point {
            // Entering a `ControlRegion` enters its first `ControlNode` child,
            // or exits the region right away (if it has no children).
            CfgPoint::RegionEntry(region) => Some(CfgCursor {
                point: match self.at(region).def().children.iter().first {
                    Some(first_child) => CfgPoint::ControlNodeEntry(first_child),
                    None => CfgPoint::RegionExit(region),
                },
                parent: cursor.parent,
            }),

            // Exiting a `ControlRegion` exits its parent `ControlNode`.
            CfgPoint::RegionExit(_) => cursor.parent.map(|parent| match parent.point {
                ControlParent::Region(_) => unreachable!(),
                ControlParent::ControlNode(parent_control_node) => CfgCursor {
                    point: CfgPoint::ControlNodeExit(parent_control_node),
                    parent: parent.parent,
                },
            }),

            // Entering a `ControlNode` depends entirely on the `ControlNodeKind`.
            CfgPoint::ControlNodeEntry(control_node) => match self.at(control_node).def().kind {
                ControlNodeKind::Block { .. } => Some(CfgCursor {
                    point: CfgPoint::ControlNodeExit(control_node),
                    parent: cursor.parent,
                }),

                ControlNodeKind::Select { .. } | ControlNodeKind::Loop { .. } => None,
            },

            // Exiting a `ControlNode` chains to a sibling/parent.
            CfgPoint::ControlNodeExit(control_node) => {
                Some(match self.control_nodes[control_node].next_in_list() {
                    // Enter the next sibling in the `ControlRegion`, if one exists.
                    Some(next_control_node) => CfgCursor {
                        point: CfgPoint::ControlNodeEntry(next_control_node),
                        parent: cursor.parent,
                    },

                    // Exit the parent `ControlRegion`.
                    None => {
                        let parent = cursor.parent.unwrap();
                        match cursor.parent.unwrap().point {
                            ControlParent::Region(parent_region) => CfgCursor {
                                point: CfgPoint::RegionExit(parent_region),
                                parent: parent.parent,
                            },
                            ControlParent::ControlNode(_) => unreachable!(),
                        }
                    }
                })
            }
        }
    }
}

impl<'a> FuncAt<'a, ControlRegion> {
    /// Traverse every [`CfgPoint`] (deeply) contained in this [`ControlRegion`],
    /// in reverse post-order (RPO), with `f` receiving each [`CfgPoint`]
    /// in turn (wrapped in [`CfgCursor`], for further traversal flexibility).
    ///
    /// RPO iteration over a CFG provides certain guarantees, most importantly
    /// that SSA definitions are visited before any of their uses.
    fn rev_post_order_for_each(self, mut f: impl FnMut(CfgCursor<'_>)) {
        self.rev_post_order_for_each_inner(&mut f, None);
    }

    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: Option<&CfgCursor<'_, ControlParent>>,
    ) {
        let region = self.position;
        f(CfgCursor {
            point: CfgPoint::RegionEntry(region),
            parent,
        });
        for func_at_control_node in self.at_children() {
            func_at_control_node.rev_post_order_for_each_inner(
                f,
                &CfgCursor {
                    point: ControlParent::Region(region),
                    parent,
                },
            );
        }
        f(CfgCursor {
            point: CfgPoint::RegionExit(region),
            parent,
        });
    }
}

impl<'a> FuncAt<'a, ControlNode> {
    fn rev_post_order_for_each_inner(
        self,
        f: &mut impl FnMut(CfgCursor<'_>),
        parent: &CfgCursor<'_, ControlParent>,
    ) {
        let child_regions: &[_] = match &self.def().kind {
            ControlNodeKind::Block { .. } => &[],
            ControlNodeKind::Select { cases, .. } => cases,
            ControlNodeKind::Loop { body, .. } => slice::from_ref(body),
        };

        let control_node = self.position;
        let parent = Some(parent);
        f(CfgCursor {
            point: CfgPoint::ControlNodeEntry(control_node),
            parent,
        });
        for &region in child_regions {
            self.at(region).rev_post_order_for_each_inner(
                f,
                Some(&CfgCursor {
                    point: ControlParent::ControlNode(control_node),
                    parent,
                }),
            );
        }
        f(CfgCursor {
            point: CfgPoint::ControlNodeExit(control_node),
            parent,
        });
    }
}

impl<'a> FuncBodyLifting<'a> {
    fn from_func_def_body(
        id_allocator: &mut IdAllocator<'_, impl FnMut() -> spv::Id>,
        func_def_body: &'a FuncDefBody,
    ) -> Self {
        let wk = &spec::Spec::get().well_known;
        let cx = id_allocator.cx;

        let mut region_inputs_source = EntityOrientedDenseMap::new();
        region_inputs_source.insert(func_def_body.body, RegionInputsSource::FuncParams);

        // Create a SPIR-V block for every CFG point needing one.
        let mut blocks = FxIndexMap::default();
        let mut visit_cfg_point = |point_cursor: CfgCursor<'_>| {
            let point = point_cursor.point;

            let phis = match point {
                CfgPoint::RegionEntry(region) => {
                    if region_inputs_source.get(region).is_some() {
                        // Region inputs handled by the parent of the region.
                        SmallVec::new()
                    } else {
                        func_def_body
                            .at(region)
                            .def()
                            .inputs
                            .iter()
                            .map(|&ControlRegionInputDecl { attrs, ty }| Phi {
                                attrs,
                                ty,

                                result_id: (id_allocator.alloc_id)(),
                                cases: FxIndexMap::default(),
                                default_value: None,
                            })
                            .collect()
                    }
                }
                CfgPoint::RegionExit(_) => SmallVec::new(),

                CfgPoint::ControlNodeEntry(control_node) => {
                    match &func_def_body.at(control_node).def().kind {
                        // The backedge of a SPIR-V structured loop points to
                        // the "loop header", i.e. the `Entry` of the `Loop`,
                        // so that's where `body` `inputs` phis have to go.
                        ControlNodeKind::Loop {
                            initial_inputs,
                            body,
                            ..
                        } => {
                            let loop_body_def = func_def_body.at(*body).def();
                            let loop_body_inputs = &loop_body_def.inputs;

                            if !loop_body_inputs.is_empty() {
                                region_inputs_source.insert(
                                    *body,
                                    RegionInputsSource::LoopHeaderPhis(control_node),
                                );
                            }

                            loop_body_inputs
                                .iter()
                                .enumerate()
                                .map(|(i, &ControlRegionInputDecl { attrs, ty })| Phi {
                                    attrs,
                                    ty,

                                    result_id: (id_allocator.alloc_id)(),
                                    cases: FxIndexMap::default(),
                                    default_value: Some(initial_inputs[i]),
                                })
                                .collect()
                        }
                        _ => SmallVec::new(),
                    }
                }
                CfgPoint::ControlNodeExit(control_node) => func_def_body
                    .at(control_node)
                    .def()
                    .outputs
                    .iter()
                    .map(|&ControlNodeOutputDecl { attrs, ty }| Phi {
                        attrs,
                        ty,

                        result_id: (id_allocator.alloc_id)(),
                        cases: FxIndexMap::default(),
                        default_value: None,
                    })
                    .collect(),
            };

            let insts = match point {
                CfgPoint::ControlNodeEntry(control_node) => {
                    match func_def_body.at(control_node).def().kind {
                        ControlNodeKind::Block { insts } => [insts].into_iter().collect(),
                        _ => SmallVec::new(),
                    }
                }
                _ => SmallVec::new(),
            };

            // Get the terminator, or reconstruct it from structured control-flow.
            let terminator = match (point, func_def_body.at(point_cursor).unique_successor()) {
                // Exiting a `ControlRegion` w/o a structured parent.
                (CfgPoint::RegionExit(region), None) => {
                    let unstructured_terminator = func_def_body
                        .unstructured_cfg
                        .as_ref()
                        .and_then(|cfg| cfg.control_inst_on_exit_from.get(region));
                    if let Some(terminator) = unstructured_terminator {
                        let cfg::ControlInst {
                            attrs,
                            kind,
                            inputs,
                            targets,
                            target_inputs,
                        } = terminator;
                        Terminator {
                            attrs: *attrs,
                            kind: Cow::Borrowed(kind),
                            reaggregated_return_value_id: match kind {
                                cfg::ControlInstKind::Return if inputs.len() > 1 => {
                                    Some((id_allocator.alloc_id)())
                                }
                                _ => None,
                            },
                            // FIXME(eddyb) borrow these whenever possible.
                            inputs: inputs.clone(),
                            targets: targets
                                .iter()
                                .map(|&target| CfgPoint::RegionEntry(target))
                                .collect(),
                            target_phi_values: target_inputs
                                .iter()
                                .map(|(&target, target_inputs)| {
                                    (CfgPoint::RegionEntry(target), &target_inputs[..])
                                })
                                .collect(),
                            merge: None,
                        }
                    } else {
                        // Structured return out of the function body.
                        assert!(region == func_def_body.body);
                        let inputs = func_def_body.at_body().def().outputs.clone();
                        Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Return),
                            reaggregated_return_value_id: if inputs.len() > 1 {
                                Some((id_allocator.alloc_id)())
                            } else {
                                None
                            },
                            inputs,
                            targets: [].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: None,
                        }
                    }
                }

                // Entering a `ControlNode` with child `ControlRegion`s.
                (CfgPoint::ControlNodeEntry(control_node), None) => {
                    let control_node_def = func_def_body.at(control_node).def();
                    match &control_node_def.kind {
                        ControlNodeKind::Block { .. } => {
                            unreachable!()
                        }

                        ControlNodeKind::Select {
                            kind,
                            scrutinee,
                            cases,
                        } => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::SelectBranch(kind.clone())),
                            reaggregated_return_value_id: None,
                            inputs: [*scrutinee].into_iter().collect(),
                            targets: cases
                                .iter()
                                .map(|&case| CfgPoint::RegionEntry(case))
                                .collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: Some(Merge::Selection(CfgPoint::ControlNodeExit(control_node))),
                        },

                        ControlNodeKind::Loop {
                            initial_inputs: _,
                            body,
                            repeat_condition: _,
                        } => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Branch),
                            reaggregated_return_value_id: None,
                            inputs: [].into_iter().collect(),
                            targets: [CfgPoint::RegionEntry(*body)].into_iter().collect(),
                            target_phi_values: FxIndexMap::default(),
                            merge: Some(Merge::Loop {
                                loop_merge: CfgPoint::ControlNodeExit(control_node),
                                // NOTE(eddyb) see the note on `Merge::Loop`'s
                                // `loop_continue` field - in particular, for
                                // SPIR-T loops, we *could* pick any point
                                // before/after/between `body`'s `children`
                                // and it should be valid *but* that had to be
                                // reverted because it's only true in the absence
                                // of divergence within the loop body itself!
                                loop_continue: CfgPoint::RegionExit(*body),
                            }),
                        },
                    }
                }

                // Exiting a `ControlRegion` to the parent `ControlNode`.
                (CfgPoint::RegionExit(region), Some(parent_exit_cursor)) => {
                    let region_outputs = Some(&func_def_body.at(region).def().outputs[..])
                        .filter(|outputs| !outputs.is_empty());

                    let parent_exit = parent_exit_cursor.point;
                    let parent_node = match parent_exit {
                        CfgPoint::ControlNodeExit(parent_node) => parent_node,
                        _ => unreachable!(),
                    };

                    match func_def_body.at(parent_node).def().kind {
                        ControlNodeKind::Block { .. } => {
                            unreachable!()
                        }

                        ControlNodeKind::Select { .. } => Terminator {
                            attrs: AttrSet::default(),
                            kind: Cow::Owned(cfg::ControlInstKind::Branch),
                            reaggregated_return_value_id: None,
                            inputs: [].into_iter().collect(),
                            targets: [parent_exit].into_iter().collect(),
                            target_phi_values: region_outputs
                                .map(|outputs| (parent_exit, outputs))
                                .into_iter()
                                .collect(),
                            merge: None,
                        },

                        ControlNodeKind::Loop {
                            initial_inputs: _,
                            body: _,
                            repeat_condition,
                        } => {
                            let backedge = CfgPoint::ControlNodeEntry(parent_node);
                            let target_phi_values = region_outputs
                                .map(|outputs| (backedge, outputs))
                                .into_iter()
                                .collect();

                            let is_infinite_loop = match repeat_condition {
                                Value::Const(cond) => {
                                    cx[cond].ctor == ConstCtor::SpvInst(wk.OpConstantTrue.into())
                                }

                                _ => false,
                            };
                            if is_infinite_loop {
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: Cow::Owned(cfg::ControlInstKind::Branch),
                                    reaggregated_return_value_id: None,
                                    inputs: [].into_iter().collect(),
                                    targets: [backedge].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            } else {
                                Terminator {
                                    attrs: AttrSet::default(),
                                    kind: Cow::Owned(cfg::ControlInstKind::SelectBranch(
                                        SelectionKind::BoolCond,
                                    )),
                                    reaggregated_return_value_id: None,
                                    inputs: [repeat_condition].into_iter().collect(),
                                    targets: [backedge, parent_exit].into_iter().collect(),
                                    target_phi_values,
                                    merge: None,
                                }
                            }
                        }
                    }
                }

                // Siblings in the same `ControlRegion` (including the
                // implied edge from a `Block`'s `Entry` to its `Exit`).
                (_, Some(succ_cursor)) => Terminator {
                    attrs: AttrSet::default(),
                    kind: Cow::Owned(cfg::ControlInstKind::Branch),
                    reaggregated_return_value_id: None,
                    inputs: [].into_iter().collect(),
                    targets: [succ_cursor.point].into_iter().collect(),
                    target_phi_values: FxIndexMap::default(),
                    merge: None,
                },

                // Impossible cases, they always return `(_, Some(_))`.
                (CfgPoint::RegionEntry(_) | CfgPoint::ControlNodeExit(_), None) => {
                    unreachable!()
                }
            };

            blocks.insert(
                point,
                BlockLifting {
                    phis,
                    insts,
                    terminator,
                },
            );
        };
        match &func_def_body.unstructured_cfg {
            None => {
                func_def_body
                    .at_body()
                    .rev_post_order_for_each(visit_cfg_point);
            }
            Some(cfg) => {
                for region in cfg.rev_post_order(func_def_body) {
                    func_def_body
                        .at(region)
                        .rev_post_order_for_each(&mut visit_cfg_point);
                }
            }
        }

        // Count the number of "uses" of each block (each incoming edge, plus
        // `1` for the entry block), to help determine which blocks are part
        // of a linear branch chain (and potentially fusable), later on.
        //
        // FIXME(eddyb) use `EntityOrientedDenseMap` here.
        let mut use_counts = FxHashMap::default();
        use_counts.reserve(blocks.len());
        let all_edges = blocks
            .first()
            .map(|(&entry_point, _)| entry_point)
            .into_iter()
            .chain(blocks.values().flat_map(|block| {
                block
                    .terminator
                    .merge
                    .iter()
                    .flat_map(|merge| {
                        let (a, b) = match merge {
                            Merge::Selection(a) => (a, None),
                            Merge::Loop {
                                loop_merge: a,
                                loop_continue: b,
                            } => (a, Some(b)),
                        };
                        [a].into_iter().chain(b)
                    })
                    .chain(&block.terminator.targets)
                    .copied()
            }));
        for target in all_edges {
            *use_counts.entry(target).or_default() += 1;
        }

        // Fuse chains of linear branches, when there is no information being
        // lost by the fusion. This is done in reverse order, so that in e.g.
        // `a -> b -> c`, `b -> c` is fused first, then when the iteration
        // reaches `a`, it sees `a -> bc` and can further fuse that into one
        // `abc` block, without knowing about `b` and `c` themselves
        // (this is possible because RPO will always output `[a, b, c]`, when
        // `b` and `c` only have one predecessor each).
        //
        // FIXME(eddyb) while this could theoretically fuse certain kinds of
        // merge blocks (mostly loop bodies) into their unique precedessor, that
        // would require adjusting the `Merge` that points to them.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for block_idx in (0..blocks.len()).rev() {
            let BlockLifting {
                terminator: original_terminator,
                ..
            } = &blocks[block_idx];

            let is_trivial_branch = {
                let Terminator {
                    attrs,
                    kind,
                    reaggregated_return_value_id,
                    inputs,
                    targets,
                    target_phi_values,
                    merge,
                } = original_terminator;

                *attrs == AttrSet::default()
                    && matches!(**kind, cfg::ControlInstKind::Branch)
                    && reaggregated_return_value_id.is_none()
                    && inputs.is_empty()
                    && targets.len() == 1
                    && target_phi_values.is_empty()
                    && merge.is_none()
            };

            if is_trivial_branch {
                let target = original_terminator.targets[0];
                let target_use_count = use_counts.get_mut(&target).unwrap();

                if *target_use_count == 1 {
                    let BlockLifting {
                        phis: ref target_phis,
                        insts: ref mut extra_insts,
                        terminator: ref mut new_terminator,
                    } = blocks[&target];

                    // FIXME(eddyb) check for block-level attributes, once/if
                    // they start being tracked.
                    if target_phis.is_empty() {
                        let extra_insts = mem::take(extra_insts);
                        let new_terminator = mem::replace(
                            new_terminator,
                            Terminator {
                                attrs: Default::default(),
                                kind: Cow::Owned(cfg::ControlInstKind::Unreachable),
                                reaggregated_return_value_id: None,
                                inputs: Default::default(),
                                targets: Default::default(),
                                target_phi_values: Default::default(),
                                merge: None,
                            },
                        );
                        *target_use_count = 0;

                        let combined_block = &mut blocks[block_idx];
                        combined_block.insts.extend(extra_insts);
                        combined_block.terminator = new_terminator;
                    }
                }
            }
        }

        // Remove now-unused blocks.
        blocks.retain(|point, _| use_counts[point] > 0);

        // Collect `OpPhi`s from other blocks' edges into each block.
        //
        // HACK(eddyb) this takes advantage of `blocks` being an `IndexMap`,
        // to iterate at the same time as mutating other entries.
        for source_block_idx in 0..blocks.len() {
            let (&source_point, source_block) = blocks.get_index(source_block_idx).unwrap();
            let targets = source_block.terminator.targets.clone();

            for target in targets {
                let source_values = {
                    let (_, source_block) = blocks.get_index(source_block_idx).unwrap();
                    source_block
                        .terminator
                        .target_phi_values
                        .get(&target)
                        .copied()
                };
                let target_block = blocks.get_mut(&target).unwrap();
                for (i, target_phi) in target_block.phis.iter_mut().enumerate() {
                    use indexmap::map::Entry;

                    let source_value = source_values
                        .map(|values| values[i])
                        .or(target_phi.default_value)
                        .unwrap();
                    match target_phi.cases.entry(source_point) {
                        Entry::Vacant(entry) => {
                            entry.insert(source_value);
                        }

                        // NOTE(eddyb) the only reason duplicates are allowed,
                        // is that `targets` may itself contain the same target
                        // multiple times (which would result in the same value).
                        Entry::Occupied(entry) => {
                            assert!(*entry.get() == source_value);
                        }
                    }
                }
            }
        }

        let mut data_insts = EntityOrientedDenseMap::new();
        let all_func_at_data_insts = blocks
            .values()
            .flat_map(|block| block.insts.iter().copied())
            .flat_map(|insts| func_def_body.at(insts));
        for func_at_inst in all_func_at_data_insts {
            let data_inst_form_def = &cx[func_at_inst.def().form];

            let mut call_spv_inst_lowering = spv::InstLowering::default();
            let spv_inst_lowering = match &data_inst_form_def.kind {
                // Disallowed while visiting.
                DataInstKind::QPtr(_) => unreachable!(),

                DataInstKind::FuncCall(callee) => {
                    if data_inst_form_def.output_types.len() > 1 {
                        call_spv_inst_lowering.disaggregated_output =
                            Some(id_allocator.ids.funcs[callee].spv_func_ret_type);
                    }
                    &call_spv_inst_lowering
                }

                DataInstKind::SpvInst(_, lowering) | DataInstKind::SpvExtInst { lowering, .. } => {
                    lowering
                }
            };

            let reaggregate_inputs = spv_inst_lowering
                .disaggregated_inputs
                .iter()
                .map(|&(_, ty)| {
                    let aggregate = id_allocator.spv_aggregate(ty);
                    let op_undef = cx.intern(ConstDef {
                        attrs: AttrSet::default(),
                        ty,
                        ctor: ConstCtor::SpvInst(wk.OpUndef.into()),
                        ctor_args: [].into_iter().collect(),
                    });
                    id_allocator.visit_const_use(op_undef);
                    let op_composite_insert_result_ids = aggregate
                        .leaves
                        .iter()
                        .map(|_| (id_allocator.alloc_id)())
                        .collect();
                    ReaggregateFromLeaves {
                        aggregate,
                        op_undef,
                        op_composite_insert_result_ids,
                    }
                })
                .collect();

            // `OpFunctionCall always has a result (but may be `OpTypeVoid`-typed).
            let has_result = matches!(data_inst_form_def.kind, DataInstKind::FuncCall(_))
                || spv_inst_lowering.disaggregated_output.is_some()
                || !data_inst_form_def.output_types.is_empty();
            let result_id = if has_result {
                Some((id_allocator.alloc_id)())
            } else {
                None
            };

            let disaggregate_result = spv_inst_lowering.disaggregated_output.map(|ty| {
                let aggregate = id_allocator.spv_aggregate(ty);
                let op_composite_extract_result_ids = aggregate
                    .leaves
                    .iter()
                    .map(|_| (id_allocator.alloc_id)())
                    .collect();
                DisaggregateToLeaves {
                    aggregate,
                    op_composite_extract_result_ids,
                }
            });

            data_insts.insert(
                func_at_inst.position,
                DataInstLifting {
                    result_id,
                    disaggregate_result,
                    reaggregate_inputs,
                },
            );
        }

        Self {
            region_inputs_source,
            data_insts,

            label_ids: blocks
                .keys()
                .map(|&point| (point, (id_allocator.alloc_id)()))
                .collect(),
            blocks,
        }
    }
}

/// Maybe-decorated "lazy" SPIR-V instruction, allowing separately emitting
/// *both* decorations (from certain [`Attr`]s), *and* the instruction itself,
/// without eagerly allocating all the instructions.
///
/// Note that SPIR-T disaggregating SPIR-V `OpTypeStruct`/`OpTypeArray`s values
/// may require additional [`spv::Inst`]s for each `LazyInst`, either for
/// reaggregating inputs, or taking apart aggregate outputs.
#[derive(Copy, Clone)]
enum LazyInst<'a, 'b> {
    Global(Global),
    OpFunction {
        func_decl: &'a FuncDecl,
        func_ids: &'b FuncIds<'a>,
    },
    OpFunctionParameter {
        param_id: spv::Id,
        param: &'a FuncParam,
    },
    OpLabel {
        label_id: spv::Id,
    },
    OpPhi {
        parent_func_ids: &'b FuncIds<'a>,
        phi: &'b Phi,
    },
    DataInst {
        parent_func_ids: &'b FuncIds<'a>,
        data_inst_def: &'a DataInstDef,
        data_inst_lifting: &'b DataInstLifting,
    },
    // FIXME(eddyb) should merge instructions be generated by `Terminator`?
    Merge(Merge<spv::Id>),
    Terminator {
        parent_func_ids: &'b FuncIds<'a>,
        terminator: &'b Terminator<'a>,
    },
    OpFunctionEnd,
}

/// [`Attr::SpvDebugLine`], extracted from [`AttrSet`], and used for emitting
/// `OpLine`/`OpNoLine` SPIR-V instructions.
#[derive(Copy, Clone, PartialEq, Eq)]
struct SpvDebugLine {
    file_path_id: spv::Id,
    line: u32,
    col: u32,
}

impl LazyInst<'_, '_> {
    fn result_id_attrs_and_import(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
    ) -> (Option<spv::Id>, AttrSet, Option<Import>) {
        let cx = module.cx_ref();

        #[allow(clippy::match_same_arms)]
        match self {
            Self::Global(global) => {
                let (attrs, import) = match global {
                    Global::Type(ty) => (cx[ty].attrs, None),
                    Global::Const(ct) => {
                        let ct_def = &cx[ct];
                        match ct_def.ctor {
                            ConstCtor::PtrToGlobalVar(gv) => {
                                let gv_decl = &module.global_vars[gv];
                                let import = match gv_decl.def {
                                    DeclDef::Imported(import) => Some(import),
                                    DeclDef::Present(_) => None,
                                };
                                (gv_decl.attrs, import)
                            }
                            ConstCtor::SpvInst { .. } => (ct_def.attrs, None),

                            // Not inserted into `globals` while visiting.
                            ConstCtor::SpvStringLiteralForExtInst(_) => unreachable!(),
                        }
                    }
                };
                (Some(ids.globals[&global]), attrs, import)
            }
            Self::OpFunction {
                func_decl,
                func_ids,
            } => {
                let import = match func_decl.def {
                    DeclDef::Imported(import) => Some(import),
                    DeclDef::Present(_) => None,
                };
                (Some(func_ids.func_id), func_decl.attrs, import)
            }
            Self::OpFunctionParameter { param_id, param } => (Some(param_id), param.attrs, None),
            Self::OpLabel { label_id } => (Some(label_id), AttrSet::default(), None),
            Self::OpPhi {
                parent_func_ids: _,
                phi,
            } => (Some(phi.result_id), phi.attrs, None),
            Self::DataInst {
                parent_func_ids: _,
                data_inst_def,
                data_inst_lifting,
            } => (data_inst_lifting.result_id, data_inst_def.attrs, None),
            Self::Merge(_) => (None, AttrSet::default(), None),
            Self::Terminator {
                parent_func_ids: _,
                terminator,
            } => (None, terminator.attrs, None),
            Self::OpFunctionEnd => (None, AttrSet::default(), None),
        }
    }

    /// Expand this `LazyInst` to one or more (see disaggregation/reaggregation
    /// note in [`LazyInst`]'s doc comment for when it can be more than one)
    /// [`spv::Inst`]s (with their respective [`SpvDebugLine`]s, if applicable),
    /// with `each_spv_inst_with_debug_line` being called for each one.
    fn for_each_spv_inst_with_debug_line(
        self,
        module: &Module,
        ids: &ModuleIds<'_>,
        mut each_spv_inst_with_debug_line: impl FnMut(spv::InstWithIds, Option<SpvDebugLine>),
    ) {
        let wk = &spec::Spec::get().well_known;
        let cx = module.cx_ref();

        let value_to_id = |parent_func_ids: &FuncIds<'_>, v| match v {
            Value::Const(ct) => match cx[ct].ctor {
                ConstCtor::SpvStringLiteralForExtInst(s) => ids.debug_strings[&cx[s]],

                _ => ids.globals[&Global::Const(ct)],
            },
            Value::ControlRegionInput { region, input_idx } => {
                let input_idx = usize::try_from(input_idx).unwrap();
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                match parent_func_body_lifting.region_inputs_source.get(region) {
                    Some(RegionInputsSource::FuncParams) => parent_func_ids.param_ids[input_idx],
                    Some(&RegionInputsSource::LoopHeaderPhis(loop_node)) => {
                        parent_func_body_lifting.blocks[&CfgPoint::ControlNodeEntry(loop_node)].phis
                            [input_idx]
                            .result_id
                    }
                    None => {
                        parent_func_body_lifting.blocks[&CfgPoint::RegionEntry(region)].phis
                            [input_idx]
                            .result_id
                    }
                }
            }
            Value::ControlNodeOutput {
                control_node,
                output_idx,
            } => {
                parent_func_ids.body.as_ref().unwrap().blocks
                    [&CfgPoint::ControlNodeExit(control_node)]
                    .phis[usize::try_from(output_idx).unwrap()]
                .result_id
            }
            Value::DataInstOutput { inst, output_idx } => {
                let output_idx = usize::try_from(output_idx).unwrap();
                let data_inst_lifting = &parent_func_ids.body.as_ref().unwrap().data_insts[inst];
                if let Some(disaggregate_result) = &data_inst_lifting.disaggregate_result {
                    disaggregate_result.op_composite_extract_result_ids[output_idx]
                } else {
                    assert_eq!(output_idx, 0);
                    data_inst_lifting.result_id.unwrap()
                }
            }
        };

        let (result_id, attrs, _) = self.result_id_attrs_and_import(module, ids);

        // FIXME(eddyb) make this less of a search and more of a
        // lookup by splitting attrs into key and value parts.
        let spv_debug_line = cx[attrs].attrs.iter().find_map(|attr| match *attr {
            Attr::SpvDebugLine {
                file_path,
                line,
                col,
            } => Some(SpvDebugLine {
                file_path_id: ids.debug_strings[&cx[file_path.0]],
                line,
                col,
            }),
            _ => None,
        });

        // HACK(eddyb) there is no need to allow `spv_debug_line` to vary per-inst.
        let mut each_inst = |inst| each_spv_inst_with_debug_line(inst, spv_debug_line);

        match self {
            Self::Global(global) => each_inst(match global {
                Global::Type(ty) => {
                    let ty_def = &cx[ty];
                    match &ty_def.ctor {
                        TypeCtor::SpvInst(inst) => spv::InstWithIds {
                            without_ids: inst.clone(),
                            result_type_id: None,
                            result_id,
                            ids: ty_def
                                .ctor_args
                                .iter()
                                .map(|&arg| {
                                    ids.globals[&match arg {
                                        TypeCtorArg::Type(ty) => Global::Type(ty),
                                        TypeCtorArg::Const(ct) => Global::Const(ct),
                                    }]
                                })
                                .collect(),
                        },

                        // Not inserted into `globals` while visiting.
                        TypeCtor::QPtr | TypeCtor::SpvStringLiteralForExtInst => unreachable!(),
                    }
                }
                Global::Const(ct) => {
                    let ct_def = &cx[ct];
                    match &ct_def.ctor {
                        &ConstCtor::PtrToGlobalVar(gv) => {
                            assert!(ct_def.attrs == AttrSet::default());
                            assert!(ct_def.ctor_args.is_empty());

                            let gv_decl = &module.global_vars[gv];

                            assert!(ct_def.ty == gv_decl.type_of_ptr_to);

                            let storage_class = match gv_decl.addr_space {
                                AddrSpace::Handles => {
                                    unreachable!(
                                        "`AddrSpace::Handles` should be legalized away before lifting"
                                    );
                                }
                                AddrSpace::SpvStorageClass(sc) => {
                                    spv::Imm::Short(wk.StorageClass, sc)
                                }
                            };
                            let initializer = match gv_decl.def {
                                DeclDef::Imported(_) => None,
                                DeclDef::Present(GlobalVarDefBody { initializer }) => initializer
                                    .map(|initializer| ids.globals[&Global::Const(initializer)]),
                            };
                            spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpVariable,
                                    imms: iter::once(storage_class).collect(),
                                },
                                result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                                result_id,
                                ids: initializer.into_iter().collect(),
                            }
                        }

                        ConstCtor::SpvInst(inst) => spv::InstWithIds {
                            without_ids: inst.clone(),
                            result_type_id: Some(ids.globals[&Global::Type(ct_def.ty)]),
                            result_id,
                            ids: ct_def
                                .ctor_args
                                .iter()
                                .map(|&ct| ids.globals[&Global::Const(ct)])
                                .collect(),
                        },

                        // Not inserted into `globals` while visiting.
                        ConstCtor::SpvStringLiteralForExtInst(_) => unreachable!(),
                    }
                }
            }),
            Self::OpFunction {
                func_decl: _,
                func_ids,
            } => {
                // FIXME(eddyb) make this less of a search and more of a
                // lookup by splitting attrs into key and value parts.
                let func_ctrl = cx[attrs]
                    .attrs
                    .iter()
                    .find_map(|attr| match *attr {
                        Attr::SpvBitflagsOperand(spv::Imm::Short(kind, word))
                            if kind == wk.FunctionControl =>
                        {
                            Some(word)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);

                each_inst(spv::InstWithIds {
                    without_ids: spv::Inst {
                        opcode: wk.OpFunction,
                        imms: iter::once(spv::Imm::Short(wk.FunctionControl, func_ctrl)).collect(),
                    },
                    result_type_id: Some(ids.globals[&Global::Type(func_ids.spv_func_ret_type)]),
                    result_id,
                    ids: iter::once(ids.globals[&Global::Type(func_ids.spv_func_type)]).collect(),
                });
            }
            Self::OpFunctionParameter { param_id: _, param } => each_inst(spv::InstWithIds {
                without_ids: wk.OpFunctionParameter.into(),
                result_type_id: Some(ids.globals[&Global::Type(param.ty)]),
                result_id,
                ids: [].into_iter().collect(),
            }),
            Self::OpLabel { label_id: _ } => each_inst(spv::InstWithIds {
                without_ids: wk.OpLabel.into(),
                result_type_id: None,
                result_id,
                ids: [].into_iter().collect(),
            }),
            Self::OpPhi {
                parent_func_ids,
                phi,
            } => each_inst(spv::InstWithIds {
                without_ids: wk.OpPhi.into(),
                result_type_id: Some(ids.globals[&Global::Type(phi.ty)]),
                result_id: Some(phi.result_id),
                ids: phi
                    .cases
                    .iter()
                    .flat_map(|(&source_point, &v)| {
                        [
                            value_to_id(parent_func_ids, v),
                            parent_func_ids.body.as_ref().unwrap().label_ids[&source_point],
                        ]
                    })
                    .collect(),
            }),
            Self::DataInst {
                parent_func_ids,
                data_inst_def,
                data_inst_lifting,
            } => {
                let DataInstFormDef { kind, output_types } = &cx[data_inst_def.form];

                let mut id_operands = SmallVec::new();

                let mut call_spv_inst_lowering = spv::InstLowering::default();
                let mut override_result_type = None;
                let (inst, spv_inst_lowering) = match kind {
                    // Disallowed while visiting.
                    DataInstKind::QPtr(_) => unreachable!(),

                    // `OpFunctionCall always has a result (but may be `OpTypeVoid`-typed).
                    DataInstKind::FuncCall(callee) => {
                        let callee_ids = &ids.funcs[callee];
                        override_result_type = Some(callee_ids.spv_func_ret_type);
                        if output_types.len() > 1 {
                            call_spv_inst_lowering.disaggregated_output = override_result_type;
                        }
                        id_operands.push(callee_ids.func_id);
                        (wk.OpFunctionCall.into(), &call_spv_inst_lowering)
                    }
                    DataInstKind::SpvInst(inst, lowering) => (inst.clone(), lowering),
                    DataInstKind::SpvExtInst {
                        ext_set,
                        inst,
                        lowering,
                    } => {
                        id_operands.push(ids.ext_inst_imports[&cx[*ext_set]]);
                        (
                            spv::Inst {
                                opcode: wk.OpExtInst,
                                imms: [spv::Imm::Short(wk.LiteralExtInstInteger, *inst)]
                                    .into_iter()
                                    .collect(),
                            },
                            lowering,
                        )
                    }
                };

                // Emit any `OpCompositeInsert`s needed by the inputs, first,
                // while gathering the `id_operands` for the instruction itself.
                let mut reaggregate_inputs = data_inst_lifting.reaggregate_inputs.iter();
                for id_operand in spv_inst_lowering.reaggreate_inputs(&data_inst_def.inputs) {
                    let value_to_id = |v| value_to_id(parent_func_ids, v);
                    let id_operand = match id_operand {
                        spv::ReaggregatedIdOperand::Direct(v) => value_to_id(v),
                        spv::ReaggregatedIdOperand::Aggregate { ty, leaves } => {
                            let result_type_id = Some(ids.globals[&Global::Type(ty)]);

                            let ReaggregateFromLeaves {
                                aggregate,
                                op_undef,
                                op_composite_insert_result_ids,
                            } = reaggregate_inputs.next().unwrap();
                            let mut aggregate_id = ids.globals[&Global::Const(*op_undef)];
                            for ((leaf, &op_composite_insert_result_id), &leaf_value) in aggregate
                                .leaves
                                .iter()
                                .zip_eq(op_composite_insert_result_ids)
                                .zip_eq(leaves)
                            {
                                each_inst(spv::InstWithIds {
                                    without_ids: leaf.op_composite_insert(),
                                    result_type_id,
                                    result_id: Some(op_composite_insert_result_id),
                                    ids: [value_to_id(leaf_value), aggregate_id]
                                        .into_iter()
                                        .collect(),
                                });
                                aggregate_id = op_composite_insert_result_id;
                            }
                            aggregate_id
                        }
                    };
                    id_operands.push(id_operand);
                }
                assert!(reaggregate_inputs.next().is_none());

                let result_type = override_result_type
                    .or(spv_inst_lowering.disaggregated_output)
                    .or_else(|| {
                        assert!(output_types.len() <= 1);
                        output_types.get(0).copied()
                    });
                each_inst(spv::InstWithIds {
                    without_ids: inst,
                    result_type_id: result_type.map(|ty| ids.globals[&Global::Type(ty)]),
                    result_id,
                    ids: id_operands,
                });

                // Emit any `OpCompositeExtract`s needed for the result, last.
                if let Some(DisaggregateToLeaves {
                    aggregate,
                    op_composite_extract_result_ids,
                }) = &data_inst_lifting.disaggregate_result
                {
                    let aggregate_id = result_id.unwrap();
                    for ((leaf, &op_composite_extract_result_id), &leaf_type) in aggregate
                        .leaves
                        .iter()
                        .zip_eq(op_composite_extract_result_ids)
                        .zip_eq(output_types)
                    {
                        each_inst(spv::InstWithIds {
                            without_ids: leaf.op_composite_extract(),
                            result_type_id: Some(ids.globals[&Global::Type(leaf_type)]),
                            result_id: Some(op_composite_extract_result_id),
                            ids: [aggregate_id].into_iter().collect(),
                        });
                    }
                }
            }
            // FIXME(eddyb) should merge instructions be generated by `Terminator`?
            Self::Merge(Merge::Selection(merge_label_id)) => each_inst(spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpSelectionMerge,
                    imms: [spv::Imm::Short(wk.SelectionControl, 0)]
                        .into_iter()
                        .collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id].into_iter().collect(),
            }),
            Self::Merge(Merge::Loop {
                loop_merge: merge_label_id,
                loop_continue: continue_label_id,
            }) => each_inst(spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpLoopMerge,
                    imms: [spv::Imm::Short(wk.LoopControl, 0)].into_iter().collect(),
                },
                result_type_id: None,
                result_id: None,
                ids: [merge_label_id, continue_label_id].into_iter().collect(),
            }),
            Self::Terminator {
                parent_func_ids,
                terminator,
            } => {
                let parent_func_body_lifting = parent_func_ids.body.as_ref().unwrap();
                let mut id_operands = terminator
                    .inputs
                    .iter()
                    .map(|&v| value_to_id(parent_func_ids, v))
                    .chain(
                        terminator
                            .targets
                            .iter()
                            .map(|&target| parent_func_body_lifting.label_ids[&target]),
                    )
                    .collect();

                if let Some(reaggregated_value_id) = terminator.reaggregated_return_value_id {
                    assert!(
                        matches!(*terminator.kind, cfg::ControlInstKind::Return)
                            && terminator.inputs.len() > 1
                    );

                    each_inst(spv::InstWithIds {
                        without_ids: wk.OpCompositeConstruct.into(),
                        result_type_id: Some(
                            ids.globals[&Global::Type(parent_func_ids.spv_func_ret_type)],
                        ),
                        result_id: Some(reaggregated_value_id),
                        ids: id_operands,
                    });
                    id_operands = [reaggregated_value_id].into_iter().collect();
                }

                let inst = match &*terminator.kind {
                    cfg::ControlInstKind::Unreachable => wk.OpUnreachable.into(),
                    cfg::ControlInstKind::Return => {
                        if terminator.inputs.is_empty() {
                            wk.OpReturn.into()
                        } else {
                            // Multiple return values get reaggregated above.
                            assert_eq!(id_operands.len(), 1);
                            wk.OpReturnValue.into()
                        }
                    }
                    cfg::ControlInstKind::ExitInvocation(cfg::ExitInvocationKind::SpvInst(
                        inst,
                    )) => inst.clone(),

                    cfg::ControlInstKind::Branch => wk.OpBranch.into(),

                    cfg::ControlInstKind::SelectBranch(SelectionKind::BoolCond) => {
                        wk.OpBranchConditional.into()
                    }
                    cfg::ControlInstKind::SelectBranch(SelectionKind::SpvInst(inst)) => {
                        inst.clone()
                    }
                };
                each_inst(spv::InstWithIds {
                    without_ids: inst,
                    result_type_id: None,
                    result_id: None,
                    ids: id_operands,
                });
            }
            Self::OpFunctionEnd => each_inst(spv::InstWithIds {
                without_ids: wk.OpFunctionEnd.into(),
                result_type_id: None,
                result_id: None,
                ids: [].into_iter().collect(),
            }),
        }
    }
}

impl Module {
    pub fn lift_to_spv_file(&self, path: impl AsRef<Path>) -> io::Result<()> {
        self.lift_to_spv_module_emitter()?.write_to_spv_file(path)
    }

    pub fn lift_to_spv_module_emitter(&self) -> io::Result<spv::write::ModuleEmitter> {
        let spv_spec = spec::Spec::get();
        let wk = &spv_spec.well_known;

        let cx = self.cx();
        let (dialect, debug_info) = match (&self.dialect, &self.debug_info) {
            (ModuleDialect::Spv(dialect), ModuleDebugInfo::Spv(debug_info)) => {
                (dialect, debug_info)
            }

            // FIXME(eddyb) support by computing some valid "minimum viable"
            // `spv::Dialect`, or by taking it as additional input.
            #[allow(unreachable_patterns)]
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "not a SPIR-V module",
                ));
            }
        };

        // Because `GlobalVar`s are given IDs by the `Const`s that point to them
        // (i.e. `ConstCtor::PtrToGlobalVar`), any `GlobalVar`s in other positions
        // require extra care to ensure the ID-giving `Const` is visited.
        let global_var_to_id_giving_global = |gv| {
            let type_of_ptr_to_global_var = self.global_vars[gv].type_of_ptr_to;
            let ptr_to_global_var = cx.intern(ConstDef {
                attrs: AttrSet::default(),
                ty: type_of_ptr_to_global_var,
                ctor: ConstCtor::PtrToGlobalVar(gv),
                ctor_args: [].into_iter().collect(),
            });
            Global::Const(ptr_to_global_var)
        };

        // Collect uses scattered throughout the module, allocating IDs for them.
        let (ids, id_bound) = {
            let mut id_bound = NonZeroUsize::new(1).unwrap();
            let mut id_allocator = IdAllocator {
                cx: &cx,
                module: self,
                alloc_id: || {
                    let id = id_bound;
                    id_bound = id_bound
                        .checked_add(1)
                        .expect("overflowing `usize` should be impossible");

                    // NOTE(eddyb) `MAX` is just a placeholder - the check for overflows
                    // is done below, after all IDs that may be allocated, have been
                    // (this is in order to not need this closure to return a `Result`).
                    id.try_into().unwrap_or(spv::Id::new(u32::MAX).unwrap())
                },
                ids: ModuleIds::default(),
                data_inst_forms_seen: FxIndexSet::default(),
                global_vars_seen: FxIndexSet::default(),
                cached_spv_aggregates: FxHashMap::default(),
            };
            id_allocator.visit_module(self);

            // See comment on `global_var_to_id_giving_global` for why this is here.
            for &gv in &id_allocator.global_vars_seen {
                id_allocator
                    .ids
                    .globals
                    .entry(global_var_to_id_giving_global(gv))
                    .or_insert_with(&mut id_allocator.alloc_id);
            }

            let ids = id_allocator.ids;

            let id_bound = spv::Id::try_from(id_bound).ok().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "ID bound of SPIR-V module doesn't fit in 32 bits",
                )
            })?;

            (ids, id_bound)
        };

        // HACK(eddyb) allow `move` closures below to reference `cx` or `ids`
        // without causing unwanted moves out of them.
        let (cx, ids) = (&*cx, &ids);

        let global_and_func_insts =
            ids.globals
                .keys()
                .copied()
                .map(LazyInst::Global)
                .chain(ids.funcs.iter().flat_map(|(&func, func_ids)| {
                    let func_decl = &self.funcs[func];
                    let body_with_lifting = match (&func_decl.def, &func_ids.body) {
                        (DeclDef::Imported(_), None) => None,
                        (DeclDef::Present(def), Some(func_body_lifting)) => {
                            Some((def, func_body_lifting))
                        }
                        _ => unreachable!(),
                    };

                    let param_insts = func_ids.param_ids.iter().zip_eq(&func_decl.params).map(
                        |(&param_id, param)| LazyInst::OpFunctionParameter { param_id, param },
                    );
                    let body_insts = body_with_lifting.map(|(func_def_body, func_body_lifting)| {
                        func_body_lifting
                            .blocks
                            .iter()
                            .flat_map(move |(point, block)| {
                                let BlockLifting {
                                    phis,
                                    insts,
                                    terminator,
                                } = block;

                                iter::once(LazyInst::OpLabel {
                                    label_id: func_body_lifting.label_ids[point],
                                })
                                .chain(phis.iter().map(|phi| LazyInst::OpPhi {
                                    parent_func_ids: func_ids,
                                    phi,
                                }))
                                .chain(
                                    insts
                                        .iter()
                                        .copied()
                                        .flat_map(move |insts| func_def_body.at(insts))
                                        .map(move |func_at_inst| {
                                            let data_inst_def = func_at_inst.def();
                                            LazyInst::DataInst {
                                                parent_func_ids: func_ids,
                                                data_inst_def,
                                                data_inst_lifting: &func_body_lifting.data_insts
                                                    [func_at_inst.position],
                                            }
                                        }),
                                )
                                .chain(terminator.merge.map(|merge| {
                                    LazyInst::Merge(match merge {
                                        Merge::Selection(merge) => {
                                            Merge::Selection(func_body_lifting.label_ids[&merge])
                                        }
                                        Merge::Loop {
                                            loop_merge,
                                            loop_continue,
                                        } => Merge::Loop {
                                            loop_merge: func_body_lifting.label_ids[&loop_merge],
                                            loop_continue: func_body_lifting.label_ids
                                                [&loop_continue],
                                        },
                                    })
                                }))
                                .chain([LazyInst::Terminator {
                                    parent_func_ids: func_ids,
                                    terminator,
                                }])
                            })
                    });

                    iter::once(LazyInst::OpFunction {
                        func_decl,
                        func_ids,
                    })
                    .chain(param_insts)
                    .chain(body_insts.into_iter().flatten())
                    .chain([LazyInst::OpFunctionEnd])
                }));

        let reserved_inst_schema = 0;
        let header = [
            spv_spec.magic,
            (u32::from(dialect.version_major) << 16) | (u32::from(dialect.version_minor) << 8),
            debug_info.original_generator_magic.map_or(0, |x| x.get()),
            id_bound.get(),
            reserved_inst_schema,
        ];

        let mut emitter = spv::write::ModuleEmitter::with_header(header);

        for cap_inst in dialect.capability_insts() {
            emitter.push_inst(&cap_inst)?;
        }
        for ext_inst in dialect.extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for (&name, &id) in &ids.ext_inst_imports {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpExtInstImport,
                    imms: spv::encode_literal_string(name).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        emitter.push_inst(&spv::InstWithIds {
            without_ids: spv::Inst {
                opcode: wk.OpMemoryModel,
                imms: [
                    spv::Imm::Short(wk.AddressingModel, dialect.addressing_model),
                    spv::Imm::Short(wk.MemoryModel, dialect.memory_model),
                ]
                .into_iter()
                .collect(),
            },
            result_type_id: None,
            result_id: None,
            ids: [].into_iter().collect(),
        })?;

        // Collect the various sources of attributes.
        let mut entry_point_insts = vec![];
        let mut execution_mode_insts = vec![];
        let mut debug_name_insts = vec![];
        let mut decoration_insts = vec![];

        for lazy_inst in global_and_func_insts.clone() {
            let (result_id, attrs, import) = lazy_inst.result_id_attrs_and_import(self, ids);

            for attr in cx[attrs].attrs.iter() {
                match attr {
                    Attr::Diagnostics(_)
                    | Attr::QPtr(_)
                    | Attr::SpvDebugLine { .. }
                    | Attr::SpvBitflagsOperand(_) => {}
                    Attr::SpvAnnotation(inst @ spv::Inst { opcode, .. }) => {
                        let target_id = result_id.expect(
                            "FIXME: it shouldn't be possible to attach \
                                 attributes to instructions without an output",
                        );

                        let inst = spv::InstWithIds {
                            without_ids: inst.clone(),
                            result_type_id: None,
                            result_id: None,
                            ids: iter::once(target_id).collect(),
                        };

                        if [wk.OpExecutionMode, wk.OpExecutionModeId].contains(opcode) {
                            execution_mode_insts.push(inst);
                        } else if [wk.OpName, wk.OpMemberName].contains(opcode) {
                            debug_name_insts.push(inst);
                        } else {
                            decoration_insts.push(inst);
                        }
                    }
                }

                if let Some(import) = import {
                    let target_id = result_id.unwrap();
                    match import {
                        Import::LinkName(name) => {
                            decoration_insts.push(spv::InstWithIds {
                                without_ids: spv::Inst {
                                    opcode: wk.OpDecorate,
                                    imms: iter::once(spv::Imm::Short(
                                        wk.Decoration,
                                        wk.LinkageAttributes,
                                    ))
                                    .chain(spv::encode_literal_string(&cx[name]))
                                    .chain([spv::Imm::Short(wk.LinkageType, wk.Import)])
                                    .collect(),
                                },
                                result_type_id: None,
                                result_id: None,
                                ids: iter::once(target_id).collect(),
                            });
                        }
                    }
                }
            }
        }

        for (export_key, &exportee) in &self.exports {
            let target_id = match exportee {
                Exportee::GlobalVar(gv) => ids.globals[&global_var_to_id_giving_global(gv)],
                Exportee::Func(func) => ids.funcs[&func].func_id,
            };
            match export_key {
                &ExportKey::LinkName(name) => {
                    decoration_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpDecorate,
                            imms: iter::once(spv::Imm::Short(wk.Decoration, wk.LinkageAttributes))
                                .chain(spv::encode_literal_string(&cx[name]))
                                .chain([spv::Imm::Short(wk.LinkageType, wk.Export)])
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id).collect(),
                    });
                }
                ExportKey::SpvEntryPoint {
                    imms,
                    interface_global_vars,
                } => {
                    entry_point_insts.push(spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpEntryPoint,
                            imms: imms.iter().copied().collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(target_id)
                            .chain(
                                interface_global_vars
                                    .iter()
                                    .map(|&gv| ids.globals[&global_var_to_id_giving_global(gv)]),
                            )
                            .collect(),
                    });
                }
            }
        }

        // FIXME(eddyb) maybe make a helper for `push_inst` with an iterator?
        for entry_point_inst in entry_point_insts {
            emitter.push_inst(&entry_point_inst)?;
        }
        for execution_mode_inst in execution_mode_insts {
            emitter.push_inst(&execution_mode_inst)?;
        }

        for (&s, &id) in &ids.debug_strings {
            emitter.push_inst(&spv::InstWithIds {
                without_ids: spv::Inst {
                    opcode: wk.OpString,
                    imms: spv::encode_literal_string(s).collect(),
                },
                result_type_id: None,
                result_id: Some(id),
                ids: [].into_iter().collect(),
            })?;
        }
        for (lang, sources) in &debug_info.source_languages {
            let lang_imms = || {
                [
                    spv::Imm::Short(wk.SourceLanguage, lang.lang),
                    spv::Imm::Short(wk.LiteralInteger, lang.version),
                ]
                .into_iter()
            };
            if sources.file_contents.is_empty() {
                emitter.push_inst(&spv::InstWithIds {
                    without_ids: spv::Inst {
                        opcode: wk.OpSource,
                        imms: lang_imms().collect(),
                    },
                    result_type_id: None,
                    result_id: None,
                    ids: [].into_iter().collect(),
                })?;
            } else {
                for (&file, contents) in &sources.file_contents {
                    // The maximum word count is `2**16 - 1`, the first word is
                    // taken up by the opcode & word count, and one extra byte is
                    // taken up by the nil byte at the end of the LiteralString.
                    const MAX_OP_SOURCE_CONT_CONTENTS_LEN: usize = (0xffff - 1) * 4 - 1;

                    // `OpSource` has 3 more operands than `OpSourceContinued`,
                    // and each of them take up exactly one word.
                    const MAX_OP_SOURCE_CONTENTS_LEN: usize =
                        MAX_OP_SOURCE_CONT_CONTENTS_LEN - 3 * 4;

                    let (contents_initial, mut contents_rest) =
                        contents.split_at(contents.len().min(MAX_OP_SOURCE_CONTENTS_LEN));

                    emitter.push_inst(&spv::InstWithIds {
                        without_ids: spv::Inst {
                            opcode: wk.OpSource,
                            imms: lang_imms()
                                .chain(spv::encode_literal_string(contents_initial))
                                .collect(),
                        },
                        result_type_id: None,
                        result_id: None,
                        ids: iter::once(ids.debug_strings[&cx[file]]).collect(),
                    })?;

                    while !contents_rest.is_empty() {
                        // FIXME(eddyb) test with UTF-8! this `split_at` should
                        // actually take *less* than the full possible size, to
                        // avoid cutting a UTF-8 sequence.
                        let (cont_chunk, rest) = contents_rest
                            .split_at(contents_rest.len().min(MAX_OP_SOURCE_CONT_CONTENTS_LEN));
                        contents_rest = rest;

                        emitter.push_inst(&spv::InstWithIds {
                            without_ids: spv::Inst {
                                opcode: wk.OpSourceContinued,
                                imms: spv::encode_literal_string(cont_chunk).collect(),
                            },
                            result_type_id: None,
                            result_id: None,
                            ids: [].into_iter().collect(),
                        })?;
                    }
                }
            }
        }
        for ext_inst in debug_info.source_extension_insts() {
            emitter.push_inst(&ext_inst)?;
        }
        for debug_name_inst in debug_name_insts {
            emitter.push_inst(&debug_name_inst)?;
        }
        for mod_proc_inst in debug_info.module_processed_insts() {
            emitter.push_inst(&mod_proc_inst)?;
        }

        for decoration_inst in decoration_insts {
            emitter.push_inst(&decoration_inst)?;
        }

        let mut current_debug_line = None;
        let mut current_block_id = None; // HACK(eddyb) for `current_debug_line` resets.
        for lazy_inst in global_and_func_insts {
            let mut result: Result<(), _> = Ok(());
            lazy_inst.for_each_spv_inst_with_debug_line(self, ids, |inst, new_debug_line| {
                if result.is_err() {
                    return;
                }

                // Reset line debuginfo when crossing/leaving blocks.
                let new_block_id = if inst.opcode == wk.OpLabel {
                    Some(inst.result_id.unwrap())
                } else if inst.opcode == wk.OpFunctionEnd {
                    None
                } else {
                    current_block_id
                };
                if current_block_id != new_block_id {
                    current_debug_line = None;
                }
                current_block_id = new_block_id;

                // Determine whether to emit `OpLine`/`OpNoLine` before `inst`,
                // in order to end up with the expected line debuginfo.
                if current_debug_line != new_debug_line {
                    let (opcode, imms, ids) = match new_debug_line {
                        Some(SpvDebugLine {
                            file_path_id,
                            line,
                            col,
                        }) => (
                            wk.OpLine,
                            [
                                spv::Imm::Short(wk.LiteralInteger, line),
                                spv::Imm::Short(wk.LiteralInteger, col),
                            ]
                            .into_iter()
                            .collect(),
                            iter::once(file_path_id).collect(),
                        ),
                        None => (
                            wk.OpNoLine,
                            [].into_iter().collect(),
                            [].into_iter().collect(),
                        ),
                    };
                    result = emitter.push_inst(&spv::InstWithIds {
                        without_ids: spv::Inst { opcode, imms },
                        result_type_id: None,
                        result_id: None,
                        ids,
                    });
                    if result.is_err() {
                        return;
                    }
                }
                current_debug_line = new_debug_line;

                result = emitter.push_inst(&inst);
            });
            result?;
        }

        Ok(emitter)
    }
}
