# Need to install module before you can use this
import pandas as pd

# pandas uses *dataframes*. These are like tables.
# a dataframe holds a set of data-entries. Data is organized into *columns*, where each column holds a list of data points

# You can make a new dataframe from a dictionary,
# where each key is a column name that maps to its values
d = { "name" : [ "Ellie", "Gayatri", "Rebecca", "Rishab" ],
      "year" : [ "Junior", "Junior", "Senior", "Junior" ],
      "major" : [ "Psychology", "CS", "CS", "MechE" ]
    }

df = pd.DataFrame(d)
print(df)

# pd.DataFrame