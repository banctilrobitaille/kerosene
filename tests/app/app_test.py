import unittest

from samitorch.inputs.utils import sample_collate

from kerosene.app.app import Experiment
from kerosene.config.parsers import YamlConfigurationParser
from kerosene.config.trainers import RunConfiguration
from kerosene.config.trainers import TrainerConfiguration, ModelTrainerConfiguration
from kerosene.events import Event
from kerosene.events.handlers.visdom import PlotCustomVariables
from kerosene.loggers.visdom import PlotType
from kerosene.loggers.visdom.config import VisdomConfiguration
from kerosene.loggers.visdom.visdom import VisdomLogger
from kerosene.training.trainers import ModelTrainerFactory


class DatasetConfiguration(object):
    def __init__(self, dataset_name, path, validation_split, training_patch_size, training_patch_step,
                 validation_patch_size, validation_patch_step, test_patch_size, test_patch_step):
        super(DatasetConfiguration, self).__init__()

        self._dataset_name = dataset_name
        self._path = path
        self._validation_split = validation_split
        self._training_patch_size = training_patch_size
        self._training_patch_step = training_patch_step
        self._validation_patch_size = validation_patch_size
        self._validation_patch_step = validation_patch_step
        self._test_patch_size = test_patch_size
        self._test_patch_step = test_patch_step

    @property
    def dataset_name(self):
        return self._dataset_name

    @property
    def path(self):
        """
        Dataset's path.

        Returns:
            str: The dataset's path.
        """
        return self._path

    @property
    def validation_split(self) -> int:
        """
        The number of validation samples.

        Returns:
            int: The number of samples in validation set.
        """
        return self._validation_split

    @property
    def training_patch_size(self):
        """
        The size of a training patch.

        Returns:
            int: The size of a training patch size.
        """
        return self._training_patch_size

    @property
    def training_patch_step(self):
        """
        The step between two training patches.

        Returns:
            int: The step.
        """
        return self._training_patch_step

    @property
    def validation_patch_size(self):
        """
        The size of a validation patch.

        Returns:
            int: The size of a validation patch size.
        """
        return self._validation_patch_size

    @property
    def validation_patch_step(self):
        """
        The step between two validation patches.

        Returns:
            int: The step.
        """
        return self._validation_patch_step

    @property
    def test_patch_size(self):
        return self._test_patch_size

    @property
    def test_patch_step(self):
        return self._test_patch_step

    @classmethod
    def from_dict(cls, dataset_name, config_dict):
        return cls(dataset_name, config_dict["path"], config_dict["validation_split"],
                   config_dict["training"]["patch_size"], config_dict["training"]["step"],
                   config_dict["validation"]["patch_size"], config_dict["validation"]["step"],
                   config_dict["test"]["patch_size"], config_dict["test"]["step"])

    def to_html(self):
        configuration_values = '\n'.join("<p>%s: %s</p>" % item for item in vars(self).items())
        return "<h2>Dataset Configuration</h2> \n {}".format(configuration_values)


class DatasetFactory(object):
    pass


class test(unittest.TestCase):

    def setUp(self) -> None:
        self._config_file = "app/config.yaml"
        self._args = {"use_amp": True, "amp_opt_level": "01", "local_rank": 0, "num_workers": 1}

    def test_app_builder(self):
        events = list()
        visdom_config = VisdomConfiguration.from_yml(self._config_file)
        visdom_logger = VisdomLogger(visdom_config)

        run_config = RunConfiguration(**self._args)
        training_config = TrainerConfiguration.from_dict(
            YamlConfigurationParser.parse_section(self._config_file, "training"))
        dataset_config = YamlConfigurationParser.parse_section(self._config_file, "dataset")
        dataset_configs = list(map(lambda dataset_name: DatasetConfiguration.from_dict(dataset_name,
                                                                                       dataset_config[dataset_name]),
                                   dataset_config))
        model_trainer_config = YamlConfigurationParser.parse_section(self._config_file, "models")
        model_trainer_configs = list(map(lambda model_name: ModelTrainerConfiguration.from_dict(model_name,
                                                                                                model_trainer_config[
                                                                                                    model_name]),
                                         model_trainer_config))

        myEvent = (PlotCustomVariables(visdom_logger, "Confusion Matrix", PlotType.HEATMAP_PLOT,
                                       params={"opts": {"columnnames": ["VM", "GM", "CSF", "Background"],
                                                        "rownames": ["Background", "CSF", "GM", "WM"],
                                                        "title": "Confusion Matrix"}},
                                       every=1), Event.ON_EPOCH_END)

        event_handlers = [myEvent]

        exp = Experiment.ExperimentBuilder() \
            .with_name("Test") \
            .with_running_config(run_config) \
            .with_training_config(training_config) \
            .with_dataset_config(dataset_configs) \
            .with_model_trainer_configs(model_trainer_configs) \
            .with_model_trainer_factory(ModelTrainerFactory) \
            .with_dataset_factory(DatasetFactory) \
            .with_collate_fn(sample_collate) \
            .with_event_handlers(event_handlers) \
            .build()
