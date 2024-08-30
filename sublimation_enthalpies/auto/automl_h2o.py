import h2o
from h2o.automl import H2OAutoML


h2o.init()

 #Import a sample binary outcome train/test set into H2O

data = h2o.upload_file("data.dat")

# Identify predictors and response( “all columns, excluding y”)
#Input
#x = ["R_1","R_2"]
#x = ["m_1","m_2"]
#x = ["en_1","en_2"]
#x = ["R_1","R_2","number_atoms"]
#x = ["m_1","m_2","number_atoms"]
#x = ["en_1","en_2","number_atoms"]
#x = ["number_atoms","R_1","R_2"]
#x = ["number_atoms","m_1","m_2"]
#x = ["number_atoms","en_1","en_2"]
#x = ["number_atoms","R_1","R_2","m_1","m_2"]
#x = ["number_atoms","R_1","R_2","en_1","en_2"]
#x = ["number_atoms","m_1","m_2","en_1","en_2"]
#x = ["R_1","R_2","m_1","m_2","en_1","en_2"]
#x = ["number_atoms","R_1","R_2","m_1","m_2","en_1","en_2"]
x = ["number_atoms","R_1","R_2","m_1","m_2","en_1","en_2"]
y = "Tm"


# split into train and validation sets
print("\n\n\n==========================================================================================")
random=int(input("Random Seed:"))
print("==========================================================================================")

train, test = data.split_frame(ratios = [.8], seed = random)


# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=10, seed=1, sort_metric = "MAE")
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard/ top performing models in the AutoML Leaderboard.
lb = aml.leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
print(lb)



from sys import exit
exit(0)
