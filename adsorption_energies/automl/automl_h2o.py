import pandas as pd
import numpy
import h2o
from h2o.automl import H2OAutoML
from sklearn.metrics import mean_absolute_error


h2o.init()

 #Import a sample binary outcome train/test set into H2O

data = h2o.upload_file("data.dat")

# Identify predictors and response( “all columns, excluding y”)
#Input
#x = ['SE']
#x=['G','SE']
#x=['AN','SE']
#x=['AN','G']
#x=['AN','G','SE']
#x=['AN','IE','SE']
#x=['G','mp','SE']
#x=['AN','G','EN','IE']
#x=['AN','G','IE','SE']
#x=['AN','G','EN','SE']
#x=['AN','G','P','IE','SE']
#x=['AN','G','P','EN','SE']
#x=['AN','G','P','EN','IE','SE']
#x = ['AN','G','P','EN','density','IE','SE']
#x = ['G','P','R','EN','density','IE','SE']
#x = ['AN','G','P','R','EN','density','IE','SE']
#x = ['AN','AM','G','P','R','EN','density','IE','SE']
x = ['G','P','EN','mp','bp','hfus','density','IE','SE']
#x = ['AN','G','P','R','EN','mp','bp','hfus','density','IE','SE']
#x = ['AN','AM','G','P','R','EN','mp','bp','hfus','density','IE','SE']
y = "CH3"


# split into train and validation sets
print("\n\n\n==========================================================================================")
random=int(input("Random Seed:"))
print("==========================================================================================")

train, test = data.split_frame(ratios = [.75], seed = random)

# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=10, seed=1, sort_metric = "MAE")
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard/ top performing models in the AutoML Leaderboard.
lb = aml.leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
print(lb)


from sys import exit
exit(0)
