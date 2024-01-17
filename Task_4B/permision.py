import os
csv_filename = "xyz.csv"
lat_long_filename = "lat_long.csv"

# Check if the file can be read
if os.access(lat_long_filename, os.R_OK):
    print(f"{lat_long_filename} can be read")
else:
    print(f"{lat_long_filename} cannot be read")

# Check if the file can be written to
if os.access(csv_filename, os.W_OK):
    print(f"{csv_filename} can be written to")
else:
    print(f"{csv_filename} cannot be written to")