import pandas as pd


name_file_path = 'resources/streets.txt'
name_df = pd.read_csv(name_file_path)

unique_names = name_df.drop_duplicates(subset=['name'])

name_output_path = 'resources/unique_streets.csv'
unique_names['name'].to_csv(name_output_path, index=False, header=False)

print(f"Уникальные названия улиц сохранены в {name_output_path}")



file_path = 'resources/cities.txt'
df = pd.read_csv(file_path)

df_cleaned = df.dropna().drop_duplicates(subset=['name']).loc[df['name'].str.len() > 2]

output_path = 'resources/unique_cities.csv'
df_cleaned.to_csv(output_path, index=False, header=False)

print(f"Уникальный список городов сохранен в {output_path}")
