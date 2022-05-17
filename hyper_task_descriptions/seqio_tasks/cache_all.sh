
export PYTHONPATH=/net/nfs.cirrascale/allennlp/hamishi/hyper-task-descriptions

PROJECT=ai2-allennlp
REGION=us-central1
TASK_NAME=seqio-t0-hyper-roberta
BUCKET=gs://hamishi-dev

# offline setting only works if you have 
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 seqio_cache_tasks \
      --tasks="ag_news_classify_question_first" \
      --output_cache_dir=gs://hamishi-dev/t0_data_roberta \
      --module_import=hyper_task_descriptions.seqio_tasks.all_t0_tasks \
      --min_shards 2 \
      --alsologtostderr
      #--pipeline_options="--runner=DataflowRunner,--project=$PROJECT,--region=$REGION,--job_name=ag-news-$TASK_NAME,--staging_location=$BUCKET/binaries,--temp_location=$BUCKET/tmp,--extra_package=/net/nfs.cirrascale/allennlp/hamishi/seqio/dist/seqio-0.0.7.tar.gz,--requirements_file=beam_requirements.txt,--extra_package=/net/nfs.cirrascale/allennlp/hamishi/hyper-task-descriptions/dist/hyper_task_descriptions-0.1.0.tar.gz,--experiments=upload_graph,--machine_type=n1-highmem-8"

