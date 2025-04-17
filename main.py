from experiment import ExperimentRunner
if __name__ == "__main__":
    runner = ExperimentRunner()
    for i in range(5):
        results = runner.run_experiments(10000, 100, i+1)