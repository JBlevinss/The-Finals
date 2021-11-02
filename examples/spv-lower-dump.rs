fn main() -> std::io::Result<()> {
    match &std::env::args().collect::<Vec<_>>()[..] {
        [_, in_file] => {
            let module = spirt::Module::lower_from_spv_file(in_file)?;

            let print_operands = |operands: &[_]| {
                spirt::spv::print::OperandPrinter {
                    operands: operands.iter(),
                    out: std::io::stderr().lock(),
                }
                .all_operands()
                .unwrap();
            };

            let spv_spec = spirt::spv::spec::Spec::get();
            let wk = &spv_spec.well_known;

            match &module.dialect {
                spirt::ModuleDialect::Spv(dialect) => {
                    eprintln!(
                        "SPIR-V {}.{} module:",
                        dialect.version_major, dialect.version_minor
                    );

                    if !dialect.capabilities.is_empty() {
                        eprintln!("  Capabilities:");
                        for &cap in &dialect.capabilities {
                            // HACK(eddyb) this is one shorter because
                            // `print_operands` always prints a space first.
                            eprint!("   ");
                            print_operands(&[spirt::spv::print::PrintOperand::Imm(
                                spirt::spv::Imm::Short(wk.Capability, cap),
                            )]);
                            eprintln!();
                        }
                    }

                    if !dialect.extensions.is_empty() {
                        eprintln!("  Extensions:");
                        for ext in &dialect.extensions {
                            eprintln!("    {}", ext);
                        }
                    }

                    // HACK(eddyb) this lacks a space because
                    // `print_operands` always prints a space first.
                    eprint!("  Addressing model:");
                    print_operands(&[spirt::spv::print::PrintOperand::Imm(
                        spirt::spv::Imm::Short(wk.AddressingModel, dialect.addressing_model),
                    )]);
                    eprintln!();

                    // HACK(eddyb) this lacks a space because
                    // `print_operands` always prints a space first.
                    eprint!("  Memory model:");
                    print_operands(&[spirt::spv::print::PrintOperand::Imm(
                        spirt::spv::Imm::Short(wk.MemoryModel, dialect.memory_model),
                    )]);
                    eprintln!();
                }
            }

            eprintln!("  All other instructions:");
            for top_level in &module.top_level {
                match top_level {
                    spirt::TopLevel::Misc(misc) => {
                        eprint!("    ");

                        if let Some(output) = &misc.output {
                            match *output {
                                spirt::MiscOutput::SpvResult {
                                    result_type_id,
                                    result_id,
                                } => {
                                    eprint!("%{}", result_id);
                                    if let Some(type_id) = result_type_id {
                                        eprint!(": %{}", type_id);
                                    }
                                    eprint!(" = ");
                                }
                            }
                        }

                        let name = match misc.kind {
                            spirt::MiscKind::SpvInst { opcode } => {
                                spv_spec.instructions.get_named(opcode).unwrap().0
                            }
                        };
                        eprint!("{}", name);

                        // FIXME(eddyb) try to make this a bit more ergonomic.
                        print_operands(
                            &misc
                                .inputs
                                .iter()
                                .map(|input| match *input {
                                    spirt::MiscInput::SpvImm(imm) => {
                                        spirt::spv::print::PrintOperand::Imm(imm)
                                    }
                                    spirt::MiscInput::SpvUntrackedId(id) => {
                                        spirt::spv::print::PrintOperand::IdLike(format!("%{}", id))
                                    }
                                    spirt::MiscInput::SpvExtInstImport(ref name) => {
                                        spirt::spv::print::PrintOperand::IdLike(format!(
                                            "%(OpExtInstImport {:?})",
                                            name
                                        ))
                                    }
                                })
                                .collect::<Vec<_>>(),
                        );
                        eprintln!();

                        if let Some(attrs) = &misc.attrs {
                            for attr in attrs.iter() {
                                eprint!("      ");
                                match attr {
                                    spirt::Attr::SpvEntryPoint {
                                        params,
                                        interface_ids,
                                    } => {
                                        eprint!("OpEntryPoint");

                                        // FIXME(eddyb) try to make this a bit more ergonomic.
                                        print_operands(
                                            &params
                                                .iter()
                                                .map(|&imm| {
                                                    spirt::spv::print::PrintOperand::Imm(imm)
                                                })
                                                .chain(interface_ids.iter().map(|&id| {
                                                    spirt::spv::print::PrintOperand::IdLike(
                                                        format!("%{}", id),
                                                    )
                                                }))
                                                .collect::<Vec<_>>(),
                                        );
                                        eprintln!();
                                    }
                                    spirt::Attr::SpvAnnotation { opcode, params } => {
                                        let name =
                                            spv_spec.instructions.get_named(*opcode).unwrap().0;
                                        eprint!("{}", name);

                                        // FIXME(eddyb) try to make this a bit more ergonomic.
                                        print_operands(
                                            &params
                                                .iter()
                                                .map(|&imm| {
                                                    spirt::spv::print::PrintOperand::Imm(imm)
                                                })
                                                .collect::<Vec<_>>(),
                                        );
                                        eprintln!();
                                    }
                                }
                            }
                        }

                        // HACK(eddyb) this makes attr attachment more obvious.
                        eprintln!();
                    }
                }
            }

            Ok(())
        }
        args => {
            eprintln!("Usage: {} IN", args[0]);
            std::process::exit(1);
        }
    }
}
