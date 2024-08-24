from exctractions import *
from transforms import *
from load import *


def display_data(element):
    print(element)
    print(80 * "*")


data = extract_from_csv("data/user_data.csv")
# display_data(data)


#  Delete all record is null
data = drop_record_all_nans(data)
# display_data(data)

# Fill messing value with mean
data = fillna(data)
# display_data(data)


# remove noisy
data = fill_noisy_data(data)
# display_data(data)

# change format data
data = change_data_type(data)
# display_data(data)

# one hot encoder
data = one_hot_encoder(data, ['gender'])
# display_data(data)

# label encoding
data = label_encoding(data, ['eye_color'])
# display_data(data)


# discretion sazi
data = k_bins_discretizer(data, ['age'])
# display_data(data)

# drop column
data = drop_column(data, ["name", 'gender_ male'])
# display_data(data)


# check_outlier_column_by_plots(data, ['height', 'weight', 'age'])


# Remove data outliers
data = remove_weight_outliers(data, 40, 120)
# display_data(data)


# Normalization
# data = min_max_scaler(data, ['age', 'height', 'weight', 'eye_color', 'salary', 'gender_female', 'gender_male'])
# display_data(data)


# Standardization
data = standard_scaler(data, ['age', 'height', 'weight', 'eye_color', 'salary', 'gender_female', 'gender_male'])
display_data(data)

load(data, './data/target.csv')
