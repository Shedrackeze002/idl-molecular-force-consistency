Implementation: 810d265a97ef06bb0b1c6c9d4d0ba6156cbead22.

The isolated bootstrap now orders reviewed scripts, repository and src paths
before site-packages. Dependency origins remain checked separately. Two
fresh-process regressions create an empty virtual environment containing loose
mace_loader.py or mace_reference_adapter.py shadows, then use the actual
bootstrap for each entry point. Only the reviewed project fixture executes;
the installed shadow cannot create its marker. Earlier startup-hook and
dependency-shadow controls pass with the new order.

Historical unresolved October 28 findings were checked against current source:

| Finding | Current control / verification |
| --- | --- |
| Live runtime/source admission | _approved_launch and _live_admission_check; loader refusal regressions |
| Independent execution audit | _validated_admission_chain; operational, packet, receipt and numerical-claim refusal tests |
| Surviving child cleanup | _terminate_owned_group; TERM-ignoring-child regression |
| Post-prediction snapshot | run_parity rehashes constructor-bound snapshot and checks size before saving final predictions; auditor validates final snapshot |
| Shared limits through persistence | supervisor setup/deadline/final-size checks; setup-delay, last-write and output-bound tests |
| Partial prediction preservation | _atomic_npz; failed-write preservation regression |
| Attempted accessor counts | counters increment before invocation; failed-energy regression |
| Species before construction | positive atomic numbers and exact validated species; zero-constructor tests |
| Raw array shapes | compare_predictions and independent audit check raw shapes before joins; surplus-row regression |

The complete eight-file WSL test output and exit record are captured alongside
this document with host paths redacted. This is synthetic implementation
verification; no real checkpoint loading or learned inference occurred.

The historical PROPOSED WSL proposal remains unchanged pending review of the
updated execution boundary and admission schema, following the coordinator's
request. These results do not constitute platform admission or launch approval.
