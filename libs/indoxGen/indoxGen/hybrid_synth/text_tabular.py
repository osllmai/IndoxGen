from .llm_generator import LLMGenerator
import pandas as pd


def initialize_llm_synth(generator_llm, judge_llm, columns, example_data, user_instruction,
                         real_data=None, diversity_threshold=0.7, max_diversity_failures=20, verbose=0):
    """
    Create a setup dictionary for the LLM text generator.

    :param generator_llm: Pretrained language model used for generating synthetic data
    :param judge_llm: Pretrained language model used for evaluating the generated data
    :param columns: List of columns that the LLM will generate
    :param example_data: List of example data dictionaries for the model to reference
    :param user_instruction: Instruction to guide the generation process
    :param real_data: Optional real data for comparison (used for evaluating synthetic data diversity)
    :param diversity_threshold: Threshold for diversity checking (default: 0.7)
    :param max_diversity_failures: Maximum number of diversity failures allowed (default: 20)
    :param verbose: Verbosity level for logging/tracking progress (default: 0)
    :return: Dictionary containing LLM setup
    """
    return {
        "generator_llm": generator_llm,
        "judge_llm": judge_llm,
        "columns": columns,
        "example_data": example_data,
        "user_instruction": user_instruction,
        "real_data": real_data,
        "diversity_threshold": diversity_threshold,
        "max_diversity_failures": max_diversity_failures,
        "verbose": verbose
    }


def initialize_gan_synth(input_dim, generator_layers, discriminator_layers, learning_rate, beta_1, beta_2, batch_size,
                         epochs, n_critic,
                         categorical_columns, mixed_columns, integer_columns):
    """
    Create a setup dictionary for the Tabular GAN, with error handling for missing indoxGen packages.

    :param input_dim: Dimension of the noise input
    :param generator_layers: List of units in generator layers
    :param discriminator_layers: List of units in discriminator layers
    :param learning_rate: Learning rate for training
    :param beta_1: Beta_1 value for the Adam optimizer
    :param beta_2: Beta_2 value for the Adam optimizer
    :param batch_size: Batch size for training
    :param epochs: Number of training epochs
    :param n_critic: Number of critic iterations per generator iteration
    :param categorical_columns: List of categorical columns
    :param mixed_columns: Dictionary of mixed columns (e.g., {"column_name": "positive"})
    :param integer_columns: List of integer columns
    :return: Dictionary containing GAN setup
    :raises ImportError: If neither `indoxGen_tensor` nor `indoxGen_torch` is installed.
    """

    try:
        from indoxGen_tensor import TabularGANConfig
        print("Successfully imported TabularGANConfig from indoxGen_tensor.")
    except ImportError:
        try:
            from indoxGen_torch import TabularGANConfig
            print("Successfully imported TabularGANConfig from indoxGen_torch.")
        except ImportError:
            raise ImportError(
                "Neither `indoxGen_tensor` nor `indoxGen_torch` is installed. "
                "Please install one of these packages to proceed."
            )

    gan_config = TabularGANConfig(
        input_dim=input_dim,
        generator_layers=generator_layers,
        discriminator_layers=discriminator_layers,
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        batch_size=batch_size,
        epochs=epochs,
        n_critic=n_critic
    )

    return {
        "config": gan_config,
        "categorical_columns": categorical_columns,
        "mixed_columns": mixed_columns,
        "integer_columns": integer_columns
    }


class TextTabularSynth:
    def __init__(self, tabular=None, text=None):
        """
        Initialize the HybridSynth class with setup configurations for both GAN (tabular) and LLM (text).

        :param tabular: Dictionary containing GAN setup (config, categorical_columns, mixed_columns, integer_columns)
        :param text: Dictionary containing LLM setup (generator_llm, judge_llm, columns, example_data, user_instruction, real_data, diversity_threshold, max_diversity_failures, verbose)
        """
        self.tabular_setup = tabular
        self.text_setup = text
        self.gan_trainer = None
        self.llm_generator = None

        if self.tabular_setup:
            self._setup_gan()
        if self.text_setup:
            self._setup_llm()

    def _setup_gan(self):
        """
        Private method to set up GAN (Tabular) training with error handling for missing packages.
        """
        tabular_config = self.tabular_setup['config']
        categorical_columns = self.tabular_setup['categorical_columns']
        mixed_columns = self.tabular_setup['mixed_columns']
        integer_columns = self.tabular_setup['integer_columns']

        try:
            from indoxGen_tensor import TabularGANTrainer
            print("Successfully imported TabularGANTrainer from indoxGen_tensor.")
        except ImportError:
            try:
                from indoxGen_torch import TabularGANTrainer
                print("Successfully imported TabularGANTrainer from indoxGen_torch.")
            except ImportError:
                raise ImportError(
                    "Neither `indoxGen_tensor` nor `indoxGen_torch` is installed. "
                    "Please install one of these packages to proceed."
                )

        self.gan_trainer = TabularGANTrainer(
            config=tabular_config,
            categorical_columns=categorical_columns,
            mixed_columns=mixed_columns,
            integer_columns=integer_columns
        )
        print("GAN setup completed.")

    def _setup_llm(self):
        """
        Private method to set up the LLM generator.
        """
        self.llm_generator = LLMGenerator(
            generator_llm=self.text_setup['generator_llm'],
            judge_llm=self.text_setup['judge_llm'],
            columns=self.text_setup['columns'],
            example_data=self.text_setup['example_data'],
            user_instruction=self.text_setup['user_instruction'],
            real_data=self.text_setup.get('real_data'),
            diversity_threshold=self.text_setup.get('diversity_threshold', 0.5),
            max_diversity_failures=self.text_setup.get('max_diversity_failures', 20),
            verbose=self.text_setup.get('verbose', 0)
        )
        print("LLM generator setup completed.")

    def generate(self, num_samples):
        """
        Generate both synthetic tabular and text data, combining training and generation steps.

        :param num_samples: Number of synthetic samples to generate
        :return: DataFrame with combined generated tabular and text data
        """
        results = {}

        if self.gan_trainer:
            self.gan_trainer.train(self.tabular_setup['data'], patience=15)
            synthetic_tabular_data = self.gan_trainer.generate_samples(num_samples)
            results['tabular_data'] = pd.DataFrame(synthetic_tabular_data)
        else:
            print("No GAN setup provided.")

        if self.llm_generator:
            synthetic_text_data = self.llm_generator.generate_data(num_samples)
            results['text_data'] = synthetic_text_data
        else:
            print("No LLM setup provided.")

        # Combine tabular and text data
        if 'tabular_data' in results and 'text_data' in results:
            combined_data = pd.concat([results['tabular_data'], results['text_data']], axis=1)
        elif 'tabular_data' in results:
            combined_data = results['tabular_data']
        elif 'text_data' in results:
            combined_data = results['text_data']
        else:
            combined_data = pd.DataFrame()

        return combined_data
