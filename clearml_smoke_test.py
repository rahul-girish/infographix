from clearml import Task

task = Task.init(
    project_name="Infographix",
    task_name="ClearML Connection Test",
    task_type="testing",
    reuse_last_task_id=False,
)

logger = task.get_logger()

logger.report_scalar(
    title="connection",
    series="test",
    value=1.0,
    iteration=0,
)

print("Successfully connected to ClearML!")

task.close()