#!/usr/bin/env python

import grok
import os

parser = grok.training.add_args(model="mlp")
parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
hparams = parser.parse_args()


hparams.datadir = os.path.abspath(hparams.datadir)
hparams.logdir = os.path.abspath(hparams.logdir)
os.environ["WANDB_API_KEY"] = "Your Key Here"


print(hparams)
print(grok.training.train(hparams, "mlp"))
