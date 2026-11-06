"""The real arm provider (plan v3 section 9): it maps a model id to a restored checkpoint arm.

A provider is `provider(model_id) -> arm`. The real one restores the model through the restore fork
(`restore_fork.e3x_checkpoint_arm`) with that model's run directory, launch record, score directory and
acceptance binding from `admission.load_bindings`. The restore fork is imported when a model is first restored,
not when this module is imported, so importing it loads no model code. `restore` is a library parameter for tests
that must not restore a checkpoint; the launch entry points never expose it.
"""


def real_arm_provider(items, data_directory, pin_policy, restore=None):
    by_id = {item["id"]: item for item in items}

    def provider(model_id):
        if model_id not in by_id:
            raise ValueError(f"no accepted binding for model {model_id!r}")
        item = by_id[model_id]
        function = restore
        if function is None:
            from stage2_capped.restore_fork import e3x_checkpoint_arm
            function = e3x_checkpoint_arm
        return function(run_directory=item["run_directory"], launch_record=item["launch_record"],
                        score_directory=item["score_directory"], acceptance=item["acceptance"],
                        data_directory=data_directory, pin_policy=pin_policy)

    return provider
