import ray

class RayTaskManage:
    def __init__(self):
        self.tasks = {}  # 存储任务ID和对应的Ray任务ObjectRef

    def submit_training_task(self, training_id, future):
        self.tasks[training_id] = future
        return training_id

    def get_task_status(self, training_id):
        if training_id not in self.tasks:
            return "Not Found"
        obj_ref = self.tasks[training_id]
        ready, _ = ray.wait([obj_ref], timeout=0)
        if ready:
            try:
                ray.get(obj_ref)
                return "Completed"
            except Exception as e:
                return f"Failed: {str(e)}"
        else:
            return "Running"

    def cancel_task(self, training_id):
        if training_id in self.tasks:
            try:
                ray.cancel(self.tasks[training_id])
            except Exception as e:
                return f"Cancel Failed: {str(e)}"
            self.tasks.pop(training_id)
            return "Task Canceled"
        return "Task Not Found"