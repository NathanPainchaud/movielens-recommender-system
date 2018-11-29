# User Based Collaborative Filtering Recommender System

This program consists of a set of Python 3.X scripts that user based collaborative filtering recommender system and test it on the MovieLens data sets.

### Environment

The package requirements are all listed in the `requirements.txt` file.

### Configuration

The script allows for options to be specified directly to the CLI, in which case they will override the default values set for the program.

### Execution

The file `run.sh` provides an example (in [Fish](https://fishshell.com/) shell) of how to call the script repeatedly with different hyperparameters,
to find those that provide the best predictions over the test sets.

However, the Python script can also be called manually.
A detailed description of the script's options is provided by the command:

```bash
python collaborative_filtering_rs.py -h
```