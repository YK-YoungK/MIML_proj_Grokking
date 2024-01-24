#!/usr/bin/env python

import grok
import os
from grok.training import TrainableTransformer, TrainableLSTM, TrainableMLP

parser = grok.training.add_args()
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()

# check only one in three models is true
model_sum = sum([hparams.transformer, hparams.lstm, hparams.mlp])
if model_sum != 1:
    print("Use exactly one of --transformer, --lstm, --mlp to specify the training model.")
    exit(1)

# add model specific args
if hparams.transformer:
    parser = TrainableTransformer.add_model_specific_args(parser)
elif hparams.lstm:
    parser = TrainableLSTM.add_model_specific_args(parser)
elif hparams.mlp:
    parser = TrainableMLP.add_model_specific_args(parser)

hparams = parser.parse_args()

hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)
os.environ["WANDB_API_KEY"] = "Your Key Here"


print(hparams)
print(grok.training.train(hparams))
