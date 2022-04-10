from gru_models import discModel
from gru_models import encodedGenerator
import torch 

# Find devices available 
torch_devs = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize discriminator (no stored weights)
disc_model = discModel(in_ch=1, in_H=256, in_W=512)

# Initialize the generator
gene_model = encodedGenerator(in_ch=1, ncvar=3, stage_channels=[32]) 

# Load generator weights.
gene_model.load_state_dict(torch.load('saved_model.pth', map_location=torch_devs))

# Sample model parameters 
batch_size=2
time_horizon=3
X_prec= torch.randn((batch_size, time_horizon, 1, 64, 128)) # sample precipitation input
X_cvar= [torch.randn((batch_size, time_horizon, 1, 64, 128)) for _ in range(3)] # sample climate covariates input
X_top = torch.randn((batch_size, 1,256, 512)) # sample elevation input 

# Use generator
y_sim = gene_model(X_prec,X_cvar, X_top )

# Use discriminator
disc_out = disc_model(y_sim, X_prec)
