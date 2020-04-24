#  <img src="/icons/oil.png" width="90" vertical-align="bottom">Kerosene
> Kerosene is a high-level deep Learning framework for fast and clean research development with Pytorch - <b>[see the doc for more details.](https://kerosene.readthedocs.io/en/latest/)</b>. Kerosene let you focus on your model and data by providing clean and readable code for training, visualizing and debugging your achitecture without forcing you to implement rigid interface for your model.

## Out of The Box Features
- [X] Basic training logic and user defined trainers
- [X] Fine grained event system with multiple handlers
- [X] Multiple metrics and criterions support
- [X] Automatic configuration parsing and model instantiation
- [X] Automatic support of mixed precision with <b>[Apex](https://github.com/NVIDIA/apex)</b> and dataparallel training
- [X] Automatic Visdom logging
- [X] Integrated <b>[Ignite](https://github.com/pytorch/ignite)</b> metrics and <b>[Pytorch](https://github.com/pytorch/pytorch)</b> criterions

## MNIST Example
 > Here is a simple example that shows how easy and clean it is to train a simple network. In very few lines of code, the model is trained using mixed precision and you got Visdom + Console logging automatically. See full example there: [MNIST-Kerosene](https://github.com/banctilrobitaille/kerosene-mnist)
 
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    model_trainer_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_train, shuffle=True)

    test_loader = DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_valid, shuffle=True)

    visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(CONFIG_FILE_PATH))

    # Initialize the model trainers
    model_trainer = ModelTrainerFactory(model=SimpleNet()).create(model_trainer_config)

    # Train with the training strategy
    SimpleTrainer("MNIST Trainer", train_loader, test_loader, None, model_trainer, RunConfiguration(use_amp=False)) \
        .with_event_handler(PlotMonitors(every=500, visdom_logger=visdom_logger), Event.ON_BATCH_END) \
        .with_event_handler(PlotAvgGradientPerLayer(every=500, visdom_logger=visdom_logger), Event.ON_TRAIN_BATCH_END) \
        .with_event_handler(PrintTrainingStatus(every=100), Event.ON_BATCH_END) \
        .train(training_config.nb_epochs)
```

## Events

| Event  | Description |
| ------------- | ------------- |
| ON_TRAINING_BEGIN  | At the beginning of the training phase  |
| ON_TRAINING_END  | At the end of the training phase  |
| ON_VALID_BEGIN  | At the beginning of the validation phase  |
| ON_VALID_END   | At the end of the validation phase  |
| ON_TEST_BEGIN  | At the beginning of the test phase  |
| ON_TEST_END   | At the end of the test phase  |
| ON_EPOCH_BEGIN  | At the beginning of each epoch (training, validation, test)   |
| ON_EPOCH_END   | At the end of each epoch (training, validation, test)   |
| ON_TRAIN_EPOCH_BEGIN   | At the beginning of each training epoch |
| ON_TRAIN_EPOCH_END   | At the end of each training epoch  |
| ON_VALID_EPOCH_BEGIN   | At the beginning of each validation epoch   |
| ON_VALID_EPOCH_END   | At the end of each validation epoch  |
| ON_TEST_EPOCH_BEGIN   | At the beginning of each test epoch   |
| ON_TEST_EPOCH_END | At the end of each test epoch  |
| ON_BATCH_BEGIN   | At the beginning of each batch (training, validation, test)  |
| ON_BATCH_END  | At the end of each batch (training, validation, test)   |
| ON_TRAIN_BATCH_BEGIN   | At the beginning of each train batch   |
| ON_TRAIN_BATCH_END   | At the end of each train batch  |
| ON_VALID_BATCH_BEGIN  | At the beginning of each validation batch   |
| ON_VALID_BATCH_END  | At the end of each validation batch  |
| ON_TEST_BATCH_BEGIN   | At the beginning of each test batch   |
| ON_TEST_BATCH_END   | At the end of each test batch  |
| ON_FINALIZE   | Before the end of the process  |

## Handlers
- [X] PrintTrainingStatus (Console)
- [X] PrintMonitors (Console)
- [X] PlotMonitors (Visdom)
- [X] PlotLosses (Visdom)
- [X] PlotMetrics (Visdom)
- [X] PlotCustomVariables (Visdom)
- [X] PlotLR (Visdom)
- [X] PlotAvgGradientPerLayer (Visdom)
- [X] Checkpoint 
- [X] EarlyStopping

## Contributing

#### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

#### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]

##### Merging code:
> Y Merged [Short Description] [Issue Number]


Icons made by <a href="http://www.flaticon.com/authors/freepik" title="Freepik">Freepik</a> from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a>
