from collections import defaultdict


class MetricMonitor:
    def __init__(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def get_metrics(self):
        return {metric_name: metric["avg"] for metric_name, metric in self.metrics.items()}