from mas_arena.agent_flow.workflow_runner import WorkflowRunner


class EvaluationUtils:

    def __init__(self, root_path: str):
        self.root_path = root_path

    async def evaluate_graph_async(self, optimizer, validation_n, data, initial=False, train_size=40, test_size=20):

        workflow_runner = WorkflowRunner(agent=optimizer.executor_agent)
        sum_score = 0

        for _ in range(validation_n):

            score, avg_cost, total_cost, all_failed = await workflow_runner.graph_evaluate_async(optimizer.evaluator,
                                                                                                 optimizer.graph,
                                                                                                 is_test=False,
                                                                                                 train_size=train_size,
                                                                                                 test_size=test_size)
            cur_round = optimizer.round + 1 if initial is False else optimizer.round
            new_data = optimizer.data_utils.create_result_data(cur_round, score, avg_cost, total_cost)
            data.append(new_data)

            result_path = optimizer.data_utils.get_results_file_path(self.root_path)
            optimizer.data_utils.save_results(result_path, data)

            sum_score += score

            if all_failed:
                print(f"All test cases failed in round {cur_round}. Stopping evaluation for this round.")
                break

        return sum_score / validation_n

    async def evaluate_graph_test_async(self, optimizer):

        evaluator = WorkflowRunner(agent=optimizer.executor_agent)

        score, avg_cost, total_cost, all_failed = await evaluator.graph_evaluate_async(optimizer.evaluator,
                                                                                       optimizer.graph,
                                                                                       is_test=True,
                                                                                       train_size=optimizer.train_size,
                                                                                       test_size=optimizer.test_size)
        return score, avg_cost, total_cost