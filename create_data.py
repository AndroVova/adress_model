import pandas as pd
import numpy as np
import random
import csv
from tqdm import tqdm
from multiprocessing import Pool

unique_streets_df = pd.read_csv('resources/Street_Names (1).csv', usecols=['street_name'])
# unique_cities_df = pd.read_csv('resources/unique_cities.csv')
cleaned_geonames_postal_code_df = pd.read_csv('resources/cleaned_geonames_postal_code.csv', usecols=[1, 2])
unique_names_df = pd.read_csv('resources/unique_names.csv', header=None, names=['name'])
# unique_companies_df = pd.read_csv('resources/German_companies_names.csv')
companies_df = pd.read_csv('resources/companies_names.csv')
# occupations_df = pd.read_csv('resources/occupation_names.csv')  
countries_df = pd.read_csv('resources/countries_list.csv')

german_english_floors = [
    "3. OG", "App. 4", "1. Stock", "2. OG", "4. Stock", "EG", "Dachgeschoss", 
    "Keller", "Souterrain", "Penthouse", "5. OG", "App. 12", "DG", "UG", 
    "3. Stock rechts", "4. OG links", "3rd Floor", "Apt. 4", "1st Floor", "2nd Floor", "4th Floor", "Ground Floor", 
    "Attic", "Basement", "Lower Ground", "Penthouse", "5th Floor", "Apt. 12", 
    "Top Floor", "Underground Floor", "3rd Floor Right", "4th Floor Left", 
    "6. OG", "7. Stock", "8. OG", "9. Stock", "10. OG", "App. 5", "6th Floor", 
    "7th Floor", "8th Floor", "9th Floor", "10th Floor", "1st Basement", "2nd Basement",
    "Mezzanine", "Lobby", "Reception", "Main Floor", "Upper Floor", "Lower Floor", 
    "Service Floor", "Garden Level", "Street Level", "Rear Floor", "Front Floor", 
    "Middle Floor", "Side Floor", "Left Wing", "Right Wing", "North Wing", 
    "South Wing", "East Wing", "West Wing", "Suite 101", "Suite 202", "Suite 303", 
    "Office 1", "Office 2", "Office 3", "Studio 1", "Studio 2", "Studio 3", 
    "Rooftop", "Terrace", "Balcony Level", "Gallery Level", "Platform Level", 
    "Cellar", "Sublevel 1", "Sublevel 2", "Annex", "Main Wing", "Side Annex", 
    "Podium", "Sky Floor", "Loft", "Atrium", "Sky Lobby", "Observation Deck"
]
prefixes = [
    "Herr", "Frau", "Dr.", "Prof.", "Herr Dr.", "Frau Dr.", 
    "Herr Prof.", "Frau Prof.", "Herr Dipl.-Ing.", "Frau Dipl.-Ing.",
    "Herr Mag.", "Frau Mag.", "Herr Ing.", "Frau Ing.", "Mr.", "Ms.",
    "Mrs.", "Mx.", "Sir", "Madam", "Lord", "Lady", "Rev.", "Capt.", 
    "Lt.", "Col.", "Gen.", "Maj.", "Brig.", "Cpt.", "Cmdr.", "Admiral", 
    "Baron", "Baroness", "Viscount", "Viscountess", "Count", "Countess", 
    "Duke", "Duchess", "Earl", "Marquis", "Marchioness", "Prince", "Princess", 
    "King", "Queen", "Emperor", "Empress", "Archduke", "Archduchess", 
    "Grand Duke", "Grand Duchess", "Sheikh", "Sheikha", "Ayatollah", 
    "Rabbi", "Pastor", "Father", "Sister", "Brother", "Monsignor", 
    "His Excellency", "Her Excellency", "Sen.", "Rep.", "Amb.", "Gov.", 
    "Hon.", "Judge", "Justice", "President", "Chancellor", "Premier", 
    "Prime Minister", "Deputy", "Councilor", "Counselor", "Attorney", 
    "Esq.", "Sheriff", "Chief", "Inspector", "Detective", "Sergeant", 
    "Officer", "Marshal", "Provost", "Principal", "Dean", "Rector", 
    "Provost", "Warden", "Magistrate", "Consul", "Envoy", "Delegate", 
    "Commander", "Supreme Leader", "Chairman", "Speaker", "Overseer"
]

positions = [
    "Abteilung Vertrieb", "Manager", "Technician", "Developer", "Projektleiter", 
    "Analyst", "Consultant", "Sales Representative", "Teamleiter", "Ingenieur", 
    "CEO", "CTO", "CFO", "Product Manager", "Support Specialist", "Marketing Coordinator",
    "Researcher", "HR Manager", "Operations Director", "Data Scientist", "Junior Developer", 
    "Middle Developer", "Senior Developer", "UX Designer", "UI Designer", "DevOps Engineer", 
    "System Administrator", "Quality Assurance", "Business Analyst", "Finance Manager", 
    "Accountant", "Legal Advisor", "Customer Service Representative", "Content Writer", 
    "Graphic Designer", "Social Media Manager", "IT Support", "Chief Marketing Officer", 
    "Logistics Coordinator", "Supply Chain Manager", "Product Owner", "Scrum Master", 
    "Software Architect", "Database Administrator", "Cybersecurity Analyst", "Mobile Developer", 
    "AI Specialist", "Machine Learning Engineer", "Network Engineer", "Electrical Engineer", 
    "Mechanical Engineer", "Civil Engineer", "Environmental Consultant", "Biomedical Engineer", 
    "Event Manager", "Training Specialist", "Talent Acquisition", "Recruiter", "Project Assistant", 
    "Warehouse Supervisor", "Field Technician", "Photographer", "Videographer", "SEO Specialist", 
    "Web Developer", "Frontend Developer", "Backend Developer", "Full Stack Developer", 
    "Operations Manager", "Account Manager", "Creative Director", "PR Specialist", 
    "Risk Manager", "Investment Analyst", "Portfolio Manager", "Copywriter", "Data Analyst", 
    "Communications Specialist", "Marketing Strategist", "Automation Engineer", "Quality Control Manager"
]

def generate_address():
    streets = unique_streets_df['street_name'].values
    house_number = str(np.random.randint(1, 101) if random.random() < 0.5 else np.random.randint(1, 11))
    if random.random() < 0.4:
        house_number += random.choice('aaaabbbbccccdddefg')
    floor_app = random.choice(german_english_floors) #todo
    city_data = cleaned_geonames_postal_code_df.sample(1).values[0]
    postal_code, city = str(city_data[0]), city_data[1]
    street_address = f"{random.choice(streets)} {house_number} {floor_app}"
    return street_address, postal_code, city

def generate_name():
    try:
        prefix = random.choice(prefixes) if random.random() < 0.3 else ""
        name = random.choice(unique_names_df['name'].values)
        surname = random.choice(unique_names_df['name'].values)
        full_name = f"{prefix} {name.title()} {surname.title()}" if prefix else f"{name.title()} {surname.title()}"
        return full_name
    except Exception:
        return "Max Verstappen"

def add_experience_level(position):
    experience_levels = ["Junior", "Middle", "Senior"]
    if random.random() < 0.3:
        level = random.choice(experience_levels)
        if not any(level in position for level in experience_levels):
            position = f"{level} {position}"
    return position

def generate_full_address():
    name = generate_name()
    company = random.choice(companies_df.sample(1)['company_name'].values) if random.random() < 0.5 else random.choice(companies_df.sample(1)['short_name'].values)
    position = add_experience_level(random.choice(positions))
    country = random.choice(countries_df['сountry'].values)
    street_address, postal_code, city = generate_address()
    return Address(name, company, position, country, street_address, postal_code, city)

class Address:
    def __init__(self, name, company, position, country, street_address, postal_code, city):
        self.name = name
        self.company = company
        self.position = position
        self.country = country
        self.street_address = street_address
        self.postal_code = postal_code
        self.city = city
    
    def get_address_formats(self):
        formats = [
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.company}\n{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.company}\n{self.position} {self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.company}\n{self.position} {self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.name}\n{self.position}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.position}\n{self.street_address}\n{self.postal_code} {self.city}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}\n{self.country}",
            f"{self.name}\n{self.street_address}\n{self.postal_code} {self.city}",
        ]
        return formats

    def get_random_format(self):
        return random.choice(self.get_address_formats())

def process_sample(_):
    address_obj = generate_full_address()
    full_address = address_obj.get_random_format()
    full_address = full_address.replace('{', '{{').replace('}', '}}')
    template = "{address}"
    text = template.format(address=full_address)
    
    tokens = text.split()
    labels = ['OOV'] * len(tokens)

    for idx, token in enumerate(tokens):
        if token in address_obj.name.split():
            labels[idx] = 'PER'
        elif token in address_obj.company.split():
            labels[idx] = 'COMP'
        elif token in address_obj.position.split():
            labels[idx] = 'POS'
        elif token == address_obj.postal_code:
            labels[idx] = 'ZIP'
        elif token in address_obj.street_address.split():
            labels[idx] = 'STREET'
        elif token in address_obj.city.split():
            labels[idx] = 'CITY'
        elif token in address_obj.country.split():
            labels[idx] = 'COUNTRY'
    
    return [text, ' '.join(labels)]

def create_labeled_data_with_tokens(num_samples=100000, file_path="resources/labeled_data.csv"):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["text", "labels"])
        with Pool() as pool:
            for result in tqdm(pool.imap_unordered(process_sample, range(num_samples)), total=num_samples, desc="Processing samples"):
                writer.writerow(result)

    print(f"Data saved to {file_path}")

create_labeled_data_with_tokens(num_samples=150000, file_path="resources/test_labeled_data.csv")
