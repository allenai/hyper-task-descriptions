# SEQIO TASKS

This folder defines task registries used for T0, and an additional registry that can be loaded independently to allow for development without loading all of P3.

t5x expects that tasks are preprocessed and cached prior to running. While I should have a gcloud bucket setup with everything needed, you can also cache tasks yourself with the following command. The task regex will cache all matching tasks defined in the imported module. I recommend caching to a shared drive or bucket so we can reuse data as much as possible :)

```bash
seqio_cache_tasks \
   --tasks=<task regex> \
   --output_cache_dir=<output_dir> \
   --module_import=hyper_task_descriptions.seqio_tasks.all_t0_tasks
```

n.b. you'll probably have to add the repo location to your `PYTHONPATH` to be able to import the t0 task module.

I'm working on a script for running all the preprocessing and caching it.
