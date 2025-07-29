import joblib
import numpy as np

model = joblib.load("model.joblib")

coef = model.coef_
intercept = model.intercept_

# Save unquantized
joblib.dump({'coef': coef, 'intercept': intercept}, 'unquant_params.joblib')

# Min-max scale to 0–255
min_val = coef.min()
max_val = coef.max()
quant_coef = ((coef - min_val) / (max_val - min_val) * 255).astype(np.uint8)
quant_intercept = int((intercept - min_val) / (max_val - min_val) * 255)

joblib.dump({'coef': quant_coef, 'intercept': quant_intercept}, 'quant_params.joblib')
