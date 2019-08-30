#  <img src="/icons/oil.png" width="90" vertical-align="bottom">Kerosene
> Deep Learning framework for fast and clean research development with Pytorch

## MNIST Example
 > Here is a simple example that shows how easy and clean it is to train a simple network. In very few lines of code, the model is trained and you got Visdom + Console logging automatically. See full example there: [MNIST-Kerosene](https://github.com/banctilrobitaille/kerosene-mnist)
 
```python
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    CONFIG_FILE_PATH = "config.yml"

    model_trainer_config, training_config = YamlConfigurationParser.parse(CONFIG_FILE_PATH)

    train_loader = DataLoader(torchvision.datasets.MNIST('/files/', train=True, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_train, shuffle=True)

    test_loader = DataLoader(torchvision.datasets.MNIST('/files/', train=False, download=True, transform=Compose(
        [ToTensor(), Normalize((0.1307,), (0.3081,))])), batch_size=training_config.batch_size_valid, shuffle=True)

    # Initialize the loggers
    visdom_logger = VisdomLogger(VisdomConfiguration.from_yml(CONFIG_FILE_PATH))

    # Initialize the model trainers
    model_trainer = ModelTrainerFactory(model=SimpleNet()).create(model_trainer_config)

    # Train with the training strategy
    trainer = SimpleTrainer("MNIST Trainer", train_loader, test_loader, model_trainer) \
        .with_event_handler(ConsoleLogger(), Event.ON_EPOCH_END) \
        .with_event_handler(visdom_logger, Event.ON_EPOCH_END, PlotAllModelStateVariables()) \
        .train(training_config.nb_epochs)
```

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
