# import json
# import re
# from typing import List, Dict, Any, Optional
# from collections import defaultdict
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from loguru import logger
# import sys
#
# # Set up logging
# logger.remove()  # Remove the default logger
# logger.add(sys.stdout, format="<green>{level}</green>: <level>{message}</level>", level="INFO")
# logger.add(sys.stdout, format="<red>{level}</red>: <level>{message}</level>", level="ERROR")
#
#
# class TextDataGenerator:
#     """
#     A class for generating synthetic data based on example data and user instructions.
#
#     This class uses language models to generate and judge synthetic data points,
#     ensuring diversity and adherence to specified criteria.
#     """
#
#     def __init__(
#         self,
#         generator_llm,
#         judge_llm,
#         columns: List[str],
#         example_data: List[Dict[str, Any]],
#         user_instruction: str,
#         real_data: Optional[List[Dict[str, Any]]] = None,
#         diversity_threshold: float = 0.5,  # Lowered threshold for higher diversity
#         max_diversity_failures: int = 20,
#         verbose: int = 0
#     ):
#
#         """
#         Initialize the SyntheticDataGenerator.
#
#         Args:
#             generator_llm: Language model for generating data.
#             judge_llm: Language model for judging data quality.
#             columns: List of column names for the synthetic data.
#             example_data: List of example data points.
#             user_instruction: Instruction for data generation.
#             real_data: Optional list of real data points.
#             diversity_threshold: Threshold for determining data diversity.
#             max_diversity_failures: Maximum number of diversity failures before forcing acceptance.
#             verbose: Verbosity level (0 for minimal output, 1 for detailed feedback).
#         """
#         self.generator_llm = generator_llm
#         self.judge_llm = judge_llm
#         self.columns = columns
#         self.example_data = example_data
#         self.user_instruction = user_instruction
#         self.real_data = real_data or []
#         self.generated_data = []
#         self.feedback_history = []
#         self.column_stats = self._calculate_column_stats()
#         self.vectorizer = TfidfVectorizer()
#         self.diversity_threshold = diversity_threshold
#         self.max_diversity_failures = max_diversity_failures
#         self.diversity_failure_count = 0
#         self.verbose = verbose
#         self.diversity_check_window = 10  # New parameter for rolling window size
#
#     def generate_data(self, num_samples: int) -> pd.DataFrame:
#         """
#         Generate synthetic data points.
#
#         Args:
#             num_samples: Number of data points to generate.
#
#         Returns:
#             DataFrame containing the generated data.
#         """
#         attempts = 0
#         max_attempts = num_samples * 10
#
#         while len(self.generated_data) < num_samples and attempts < max_attempts:
#             attempts += 1
#             generated = self._generate_single_data_point()
#             if not generated:
#                 continue
#
#             score = self._judge_data_point(generated)
#
#             if score >= 0.6 and self._is_diverse(generated):
#                 self.generated_data.append(generated)
#                 self.diversity_failure_count = 0
#                 if self.verbose >= 1:
#                     logger.info(f"Generated data point: {generated}")
#             elif score >= 0.6:
#                 self._handle_diversity_failure(generated)
#             else:
#                 self._inform_generator(generated, score, "Low score")
#
#             if self.verbose >= 1 and attempts % 10 == 0:
#                 logger.info(
#                     f"Progress: {len(self.generated_data)}/{num_samples} data points generated. Attempts: {attempts}")
#
#         if len(self.generated_data) < num_samples:
#             logger.warning(
#                 f"Only generated {len(self.generated_data)} out of {num_samples} requested samples after {attempts} attempts.")
#
#         return self._convert_to_dataframe()
#
#     def _generate_single_data_point(self, context=None) -> Dict[str, Any]:
#         """Generate a single data point with optional context."""
#         system_prompt = (
#             "You are an advanced synthetic data generator tasked with creating unique and diverse text data. "
#             "Ensure that each output is significantly different from previous ones and from the examples provided. "
#             "Your response must be a valid JSON object with all property names enclosed in double quotes."
#         )
#         prompt = self._create_generation_prompt(context)
#
#         for attempt in range(3):
#             try:
#                 generated = self.generator_llm.chat(prompt, system_prompt=system_prompt)
#                 # Find the first '{' and last '}' to extract the JSON object
#                 start = generated.find('{')
#                 end = generated.rfind('}')
#                 if start != -1 and end != -1 and start < end:
#                     json_str = generated[start:end + 1]
#                     # Replace single quotes with double quotes for property names
#                     json_str = re.sub(r"(\w+):", r'"\1":', json_str)
#                     # Remove any control characters
#                     json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
#                     # Ensure all string values are properly quoted
#                     json_str = re.sub(r': *([^",\{\}\[\]]+)([,\}])', r': "\1"\2', json_str)
#                     data = json.loads(json_str)
#
#                     if set(self.columns).issubset(data.keys()):
#                         return data
#                     else:
#                         missing_columns = set(self.columns) - set(data.keys())
#                         if self.verbose >= 1:
#                             logger.warning(f"Generated data is missing columns: {missing_columns}")
#                 else:
#                     if self.verbose >= 1:
#                         logger.error(f"Failed to find valid JSON object in generated text (Attempt {attempt + 1}/3)")
#             except json.JSONDecodeError as e:
#                 if self.verbose >= 1:
#                     logger.error(f"Failed to parse generated data (Attempt {attempt + 1}/3): {str(e)}")
#                     logger.error(f"Problematic JSON string: {json_str}")
#
#             if self.verbose >= 1 and attempt < 2:
#                 logger.info(f"Retrying generation (Attempt {attempt + 2}/3)...")
#
#         if self.verbose >= 1:
#             logger.warning("Max attempts reached. Skipping this data point.")
#         return {}
#
#     def _calculate_column_stats(self) -> Dict[str, Dict[str, Any]]:
#         """Calculate statistics for each column in the dataset."""
#         stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'mean': 0, 'unique_values': set()})
#         all_data = self.example_data + self.real_data
#
#         for data in all_data:
#             for col, value in data.items():
#                 if isinstance(value, (int, float)):
#                     stats[col]['min'] = min(stats[col]['min'], value)
#                     stats[col]['max'] = max(stats[col]['max'], value)
#                     stats[col]['mean'] += value
#                 elif isinstance(value, str):
#                     stats[col]['unique_values'].add(value)
#
#         for col, col_stats in stats.items():
#             if 'mean' in col_stats:
#                 col_stats['mean'] /= len(all_data)
#                 col_stats['std'] = np.std([data[col] for data in all_data if isinstance(data.get(col), (int, float))])
#
#         return dict(stats)
#
#     def _create_generation_prompt(self, context=None) -> str:
#       """
#       Create a prompt for generating text that uses the numerical context.
#       """
#       prompt = f"Generate a synthetic data point with the following text columns: {', '.join(self.columns)}.\n"
#       prompt += f"User instruction: {self.user_instruction}\n"
#       prompt += (
#           "Ensure that the generated text is coherent with the provided numerical data. "
#           "Each output should be realistic and diverse, and significantly different from previous outputs.\n"
#           "Vary the language used in the 'remarks' field significantly.\n\n"
#       )
#
#       if context:
#           prompt += "Numerical context for this data point:\n"
#           for key, value in context.items():
#               prompt += f"- {key}: {value}\n"
#           prompt += "\nEnsure the generated text aligns with this numerical context.\n"
#
#       if self.example_data:
#           prompt += "\nHere are some example data points:\n"
#           for example in self.example_data[:5]:
#               example_context = {k: v for k, v in example.items() if k not in self.columns}
#               prompt += "Example Context:\n"
#               for key, value in example_context.items():
#                   prompt += f"- {key}: {value}\n"
#               prompt += f"Example Text Data: {json.dumps({col: example[col] for col in self.columns})}\n\n"
#
#       prompt += "\nGenerate a single data point as a JSON object. Ensure the text reflects the provided numerical data."
#       return prompt
#
#
#
#     def _is_diverse(self, new_data: Dict[str, Any]) -> bool:
#         """Check if the new data point is diverse compared to existing data."""
#         if len(self.generated_data) < 2:
#             return True
#
#         new_text = json.dumps(new_data)
#         existing_texts = [json.dumps(data) for data in self.generated_data[-self.diversity_check_window:]]
#
#         all_texts = existing_texts + [new_text]
#         tfidf_matrix = self.vectorizer.fit_transform(all_texts)
#         cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
#
#         # Calculate the mean similarity
#         mean_similarity = np.mean(cosine_similarities)
#
#         # Calculate the standard deviation of similarities
#         std_similarity = np.std(cosine_similarities)
#
#         # Implement a "soft" diversity check
#         if mean_similarity < self.diversity_threshold:
#             return True
#         elif mean_similarity < self.diversity_threshold + 0.1 and std_similarity > 0.1:
#             return True
#         else:
#             return False
#
#     def _handle_diversity_failure(self, generated: Dict[str, Any]):
#         """Handle the case when a generated data point fails the diversity check."""
#         self.diversity_failure_count += 1
#         if self.verbose >= 1:
#             logger.warning(
#                 f"Generated data is not diverse. Retrying... (Failure count: {self.diversity_failure_count})")
#         if self.diversity_failure_count >= self.max_diversity_failures:
#             if self.verbose >= 1:
#                 logger.info("Max diversity failures reached. Forcing acceptance of this data point.")
#             self.generated_data.append(generated)
#             self.diversity_failure_count = 0
#         elif self.diversity_failure_count % 5 == 0:
#             # Every 5 failures, slightly increase the diversity threshold
#             self.diversity_threshold += 0.05
#             if self.verbose >= 1:
#                 logger.info(f"Increased diversity threshold to {self.diversity_threshold}")
#
#     def _judge_data_point(self, data: Dict[str, Any]) -> float:
#         """Judge the quality of a generated data point."""
#         system_prompt = ("You are a data quality judge. Evaluate the given data based on the criteria and return a "
#                          "score between 0 and 1. It's important to only send score without any description")
#         criteria = self._create_judge_criteria()
#         prompt = (f"Data to evaluate: {json.dumps(data)}\n\nCriteria:\n{criteria}\n\nProvide a numeric score between 0 "
#                   f"and 1.")
#
#         score_str = self.judge_llm.chat(prompt, system_prompt=system_prompt)
#         try:
#             return float(score_str)
#         except ValueError:
#             if self.verbose >= 1:
#                 logger.error(f"Failed to parse judge score: {score_str}")
#             return 0.5
#
#     def _inform_generator(self, data: Dict[str, Any], score: float, reason: str):
#         """Inform the generator about the quality of generated data."""
#         feedback = f"Generated data: {json.dumps(data)}\nScore: {score}\nReason: {reason}"
#         self.feedback_history.append(feedback)
#         if self.verbose >= 1:
#             logger.info(f"Feedback for generator: {feedback}")
#
#     def _convert_to_dataframe(self) -> pd.DataFrame:
#         """Convert generated data to a pandas DataFrame."""
#         return pd.DataFrame(self.generated_data)
#
#     def _create_judge_criteria(self) -> str:
#         """Create criteria for judging generated data."""
#         criteria = "Evaluate the generated data based on the following criteria:\n"
#         criteria += f"1. Adheres to the user instruction: {self.user_instruction}\n"
#         criteria += "2. Contains all required columns.\n"
#         criteria += "3. Data types match the example data.\n"
#         criteria += "4. Values are plausible and make sense within the context.\n"
#         criteria += "5. Avoids clear personal information like full names, addresses.\n"
#         criteria += "6. Demonstrates significant creativity while maintaining realism.\n"
#         criteria += "7. Shows high diversity compared to previously generated data.\n"
#         criteria += ("Return a score between 0 and 1, where 1 is perfect. Only return the numeric score without any "
#                      "additional text.")
#         return criteria