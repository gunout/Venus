# Venus.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class VenusDataAnalyzer:
    def __init__(self, data_type):
        self.data_type = data_type
        self.colors = ['#FFD700', '#E6BE8A', '#DAA520', '#B8860B', '#FFDF00', 
                      '#F0E68C', '#EEE8AA', '#BDB76B', '#FFFACD', '#FFEFD5']
        
        self.start_year = 1960  # Début des observations vénusiennes sérieuses
        self.end_year = 2025
        
        # Configuration spécifique pour chaque type de données vénusiennes
        self.config = self._get_venus_config()
        
    def _get_venus_config(self):
        """Retourne la configuration spécifique pour chaque type de données vénusiennes"""
        configs = {
            "temperature": {
                "base_value": 462,
                "cycle_years": 0.62,  # Jour vénusien en années terrestres
                "amplitude": 5,
                "trend": "extreme",
                "unit": "°C",
                "description": "Température moyenne de surface"
            },
            "atmospheric_pressure": {
                "base_value": 92,
                "cycle_years": 0.62,
                "amplitude": 2,
                "trend": "stable",
                "unit": "bars",
                "description": "Pression atmosphérique en surface"
            },
            "cloud_cover": {
                "base_value": 75,
                "cycle_years": 0.62,
                "amplitude": 15,
                "trend": "permanent",
                "unit": "% de couverture",
                "description": "Couverture nuageuse d'acide sulfurique"
            },
            "surface_conditions": {
                "base_value": 100,
                "cycle_years": 0.62,
                "amplitude": 20,
                "trend": "infernal",
                "unit": "Index d'hostilité",
                "description": "Conditions de surface extrêmes"
            },
            "volcanic_activity": {
                "base_value": 25,
                "cycle_years": 8.0,
                "amplitude": 20,
                "trend": "cyclique",
                "unit": "Index volcanique",
                "description": "Activité volcanique présumée"
            },
            "solar_radiation": {
                "base_value": 200,
                "cycle_years": 11.0,
                "amplitude": 50,
                "trend": "attenue",
                "unit": "W/m²",
                "description": "Radiation solaire en surface"
            },
            "atmospheric_composition": {
                "base_value": 96.5,
                "cycle_years": 1.0,  # Changé de 0 à 1 pour éviter la division par zéro
                "amplitude": 0.1,    # Réduit car très stable
                "trend": "constant",
                "unit": "% CO₂",
                "description": "Composition atmosphérique (CO₂)"
            },
            "wind_speeds": {
                "base_value": 5,
                "cycle_years": 0.62,
                "amplitude": 300,
                "trend": "super-rotation",
                "unit": "km/h",
                "description": "Vents atmosphériques (surface vs haute atmosphère)"
            },
            "orbital_distance": {
                "base_value": 0.72,
                "cycle_years": 0.62,
                "amplitude": 0.01,
                "trend": "stable",
                "unit": "UA",
                "description": "Distance au Soleil"
            },
            # Configuration par défaut
            "default": {
                "base_value": 100,
                "cycle_years": 0.62,
                "amplitude": 20,
                "trend": "stable",
                "unit": "Unités",
                "description": "Données vénusiennes génériques"
            }
        }
        
        return configs.get(self.data_type, configs["default"])
    
    def generate_venus_data(self):
        """Génère des données vénusiennes simulées basées sur les caractéristiques uniques de Vénus"""
        print(f"♀️ Génération des données vénusiennes pour {self.config['description']}...")
        
        # Créer une base de données annuelle (en années terrestres)
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Earth_Year': [date.year for date in dates]}
        data['Venus_Day'] = self._earth_to_venus_days(dates)
        
        # Données principales basées sur les caractéristiques vénusiennes
        data['Base_Value'] = self._simulate_venus_cycle(dates)
        data['Surface_Conditions'] = self._simulate_surface_conditions(dates)
        data['Atmospheric_Effects'] = self._simulate_atmospheric_effects(dates)
        data['Solar_Day_Phase'] = self._simulate_solar_day_phase(dates)
        
        # Variations environnementales
        data['Climate_Trend'] = self._simulate_climate_trend(dates)
        data['Cloud_Variations'] = self._simulate_cloud_variations(dates)
        data['Volcanic_Influence'] = self._simulate_volcanic_influence(dates)
        
        # Données dérivées
        data['Smoothed_Value'] = self._simulate_smoothed_data(dates)
        data['Diurnal_Variation'] = self._simulate_diurnal_variation(dates)
        data['Annual_Variation'] = self._simulate_annual_variation(dates)
        
        # Indices vénusiens complémentaires
        data['Venus_Index'] = self._simulate_venus_index(dates)
        data['Hostility_Level'] = self._simulate_hostility_level(dates)
        data['Future_Prediction'] = self._simulate_future_prediction(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des événements vénusiens historiques
        self._add_venus_events(df)
        
        return df
    
    def _earth_to_venus_days(self, dates):
        """Convertit les années terrestres en jours vénusiens"""
        venus_days = []
        venus_day_duration = 0.62  # Années terrestres pour un jour vénusien
        
        for date in dates:
            earth_year = date.year
            venus_day = (earth_year - self.start_year) / venus_day_duration
            venus_days.append(venus_day)
        
        return venus_days
    
    def _simulate_venus_cycle(self, dates):
        """Simule le cycle vénusien principal (jour solaire très long)"""
        base_value = self.config["base_value"]
        cycle_years = self.config["cycle_years"]
        amplitude = self.config["amplitude"]
        
        # Protection contre la division par zéro
        if cycle_years == 0:
            # Retourner une valeur constante pour les données sans cycle
            return [base_value + np.random.normal(0, amplitude * 0.01) for _ in range(len(dates))]
        
        values = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            # Cycle diurne vénusien (0.62 années terrestres = 1 jour vénusien)
            venus_phase = (earth_year - self.start_year) % cycle_years
            diurnal_cycle = np.sin(2 * np.pi * venus_phase / cycle_years)
            
            # Effet de super-rotation atmosphérique
            super_rotation_phase = (earth_year - self.start_year) % 0.62
            super_rotation_cycle = np.cos(2 * np.pi * super_rotation_phase / 0.62)
            
            # Combinaison des cycles
            if self.config["trend"] == "extreme":
                value = base_value + amplitude * diurnal_cycle
            elif self.config["trend"] == "super-rotation":
                value = base_value + amplitude * 0.1 * diurnal_cycle + amplitude * 0.9 * super_rotation_cycle
            elif self.config["trend"] == "cyclique":
                value = base_value + amplitude * (0.7 * diurnal_cycle + 0.3 * super_rotation_cycle)
            else:
                value = base_value + amplitude * 0.2 * diurnal_cycle
            
            # Bruit environnemental vénusien
            noise = np.random.normal(0, amplitude * 0.05)
            values.append(value + noise)
        
        return values
    
    def _simulate_surface_conditions(self, dates):
        """Simule les conditions extrêmes de surface"""
        conditions = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            # Conditions de surface extrêmement stables mais hostiles
            if earth_year < 1970:
                condition = 0.9  # Période pré-exploration détaillée
            elif 1970 <= earth_year < 1980:
                condition = 1.0 + 0.01 * (earth_year - 1970)  # Missions Venera
            elif 1980 <= earth_year < 1990:
                condition = 1.1 + 0.005 * (earth_year - 1980)  # Missions Vega
            elif 1990 <= earth_year < 2000:
                condition = 1.15 + 0.003 * (earth_year - 1990)  # Magellan
            elif 2000 <= earth_year < 2010:
                condition = 1.18 + 0.002 * (earth_year - 2000)  # Venus Express
            elif 2010 <= earth_year < 2020:
                condition = 1.20 + 0.001 * (earth_year - 2010)  # Akatsuki
            else:
                condition = 1.21 + 0.0005 * (earth_year - 2020)  # Missions futures
            
            conditions.append(condition)
        
        return conditions
    
    def _simulate_atmospheric_effects(self, dates):
        """Simule les effets atmosphériques uniques de Vénus"""
        effects = []
        for date in dates:
            earth_year = date.year
            
            # Effet de serre extrême constant
            greenhouse_effect = 500  # Effet de serre extrême
            
            # Légères variations dues à l'activité solaire
            solar_phase = (earth_year - self.start_year) % 11.0
            solar_variation = 0.01 * np.sin(2 * np.pi * solar_phase / 11.0)
            
            effect = greenhouse_effect * (1 + solar_variation)
            effects.append(effect)
        
        return effects
    
    def _simulate_solar_day_phase(self, dates):
        """Simule la phase du jour solaire vénusien (0-1)"""
        phases = []
        for date in dates:
            earth_year = date.year
            phase = (earth_year - self.start_year) % 0.62 / 0.62
            phases.append(phase)
        
        return phases
    
    def _simulate_climate_trend(self, dates):
        """Simule les tendances climatiques à long terme"""
        trends = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            # Climat extrêmement stable sur Vénus
            trend = 1.0  # Très peu de variation climatique
            
            trends.append(trend)
        
        return trends
    
    def _simulate_cloud_variations(self, dates):
        """Simule les variations de la couverture nuageuse"""
        cloud_variations = []
        for date in dates:
            earth_year = date.year
            
            # Nuages permanents avec légères variations
            venus_day_phase = (earth_year - self.start_year) % 0.62
            cloud_variation = 0.1 * np.sin(2 * np.pi * venus_day_phase / 0.62)
            
            cloud_level = 1.0 + cloud_variation
            cloud_variations.append(cloud_level)
        
        return cloud_variations
    
    def _simulate_volcanic_influence(self, dates):
        """Simule l'influence de l'activité volcanique présumée"""
        volcanic_effects = []
        for date in dates:
            earth_year = date.year
            
            # Cycle volcanique hypothétique de 8 ans
            volcanic_phase = (earth_year - self.start_year) % 8.0
            volcanic_effect = 1.0 + 0.3 * np.sin(2 * np.pi * volcanic_phase / 8.0)
            
            volcanic_effects.append(volcanic_effect)
        
        return volcanic_effects
    
    def _simulate_smoothed_data(self, dates):
        """Simule des données lissées"""
        base_cycle = self._simulate_venus_cycle(dates)
        
        smoothed = []
        window_size = 3  # Années terrestres
        
        for i in range(len(base_cycle)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(base_cycle), i + window_size//2 + 1)
            window = base_cycle[start_idx:end_idx]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _simulate_diurnal_variation(self, dates):
        """Simule les variations diurnes (très faibles sur Vénus)"""
        variations = []
        for date in dates:
            earth_year = date.year
            # Variation diurne très faible due à la lente rotation
            venus_day_phase = (earth_year - self.start_year) % 0.62 / 0.62
            diurnal_variation = 0.01 * np.sin(2 * np.pi * venus_day_phase)
            variations.append(1 + diurnal_variation)
        
        return variations
    
    def _simulate_annual_variation(self, dates):
        """Simule les variations annuelles terrestres"""
        variations = []
        for i, date in enumerate(dates):
            earth_year = date.year
            annual_variation = 0.01 * np.sin(2 * np.pi * (earth_year - self.start_year) / 1.0)
            variations.append(1 + annual_variation)
        
        return variations
    
    def _simulate_venus_index(self, dates):
        """Simule un indice vénusien composite"""
        indices = []
        base_cycle = self._simulate_venus_cycle(dates)
        surface_conditions = self._simulate_surface_conditions(dates)
        cloud_variations = self._simulate_cloud_variations(dates)
        
        for i in range(len(dates)):
            # Indice composite pondéré
            index = (base_cycle[i] * 0.6 + 
                    surface_conditions[i] * 20 * 0.3 +
                    cloud_variations[i] * 10 * 0.1)
            indices.append(index)
        
        return indices
    
    def _simulate_hostility_level(self, dates):
        """Simule le niveau d'hostilité environnementale (0-100)"""
        hostility_levels = []
        surface_conditions = self._simulate_surface_conditions(dates)
        
        for condition in surface_conditions:
            # Niveau d'hostilité basé sur les conditions de surface
            hostility = min(100, (condition - 0.9) * 333)  # Échelle 0-100
            hostility_levels.append(hostility)
        
        return hostility_levels
    
    def _simulate_future_prediction(self, dates):
        """Simule des prédictions futures"""
        predictions = []
        base_cycle = self._simulate_venus_cycle(dates)
        
        for i, date in enumerate(dates):
            earth_year = date.year
            current_value = base_cycle[i]
            
            if earth_year > 2020:  # Période de prédiction
                # Très faible incertitude due à la stabilité vénusienne
                years_since_2020 = earth_year - 2020
                uncertainty = 0.01 * years_since_2020
                prediction = current_value * (1 + np.random.normal(0, uncertainty))
            else:
                prediction = current_value
            
            predictions.append(prediction)
        
        return predictions
    
    def _add_venus_events(self, df):
        """Ajoute des événements vénusiens historiques significatifs"""
        for i, row in df.iterrows():
            earth_year = row['Earth_Year']
            
            # Événements d'observation vénusienne
            if earth_year == 1962:
                # Mariner 2 - premier survol réussi de Vénus
                df.loc[i, 'Hostility_Level'] = 10
            
            elif 1967 <= earth_year <= 1969:
                # Venera 4, 5, 6 - premières entrées atmosphériques
                df.loc[i, 'Hostility_Level'] = 30
            
            elif earth_year == 1970:
                # Venera 7 - premier atterrissage réussi
                df.loc[i, 'Surface_Conditions'] *= 1.05
                df.loc[i, 'Hostility_Level'] = 50
            
            elif earth_year == 1975:
                # Venera 9 et 10 - premières images de surface
                df.loc[i, 'Surface_Conditions'] *= 1.1
                df.loc[i, 'Hostility_Level'] = 70
            
            elif earth_year == 1978:
                # Pioneer Venus - étude atmosphérique complète
                df.loc[i, 'Hostility_Level'] = 60
            
            elif earth_year == 1982:
                # Venera 13 et 14 - atterrissages avancés
                df.loc[i, 'Surface_Conditions'] *= 1.15
                df.loc[i, 'Hostility_Level'] = 80
            
            elif earth_year == 1985:
                # Vega 1 et 2 - ballons atmosphériques
                df.loc[i, 'Hostility_Level'] = 65
            
            elif earth_year == 1990:
                # Magellan - cartographie radar
                df.loc[i, 'Hostility_Level'] = 75
            
            elif earth_year == 2005:
                # Venus Express - étude atmosphérique
                df.loc[i, 'Hostility_Level'] = 70
            
            elif earth_year == 2010:
                # Akatsuki - étude climatique
                df.loc[i, 'Hostility_Level'] = 72
    
    def create_venus_analysis(self, df):
        """Crée une analyse complète des données vénusiennes"""
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Cycle vénusien principal
        ax1 = plt.subplot(5, 2, 1)
        self._plot_venus_cycle(df, ax1)
        
        # 2. Conditions de surface
        ax2 = plt.subplot(5, 2, 2)
        self._plot_surface_conditions(df, ax2)
        
        # 3. Variations diurnes
        ax3 = plt.subplot(5, 2, 3)
        self._plot_diurnal_variations(df, ax3)
        
        # 4. Effets atmosphériques
        ax4 = plt.subplot(5, 2, 4)
        self._plot_atmospheric_effects(df, ax4)
        
        # 5. Phase du jour solaire
        ax5 = plt.subplot(5, 2, 5)
        self._plot_solar_day_phase(df, ax5)
        
        # 6. Données lissées
        ax6 = plt.subplot(5, 2, 6)
        self._plot_smoothed_data_plot(df, ax6)
        
        # 7. Niveau d'hostilité
        ax7 = plt.subplot(5, 2, 7)
        self._plot_hostility_level_plot(df, ax7)
        
        # 8. Variations nuageuses
        ax8 = plt.subplot(5, 2, 8)
        self._plot_cloud_variations(df, ax8)
        
        # 9. Indice vénusien
        ax9 = plt.subplot(5, 2, 9)
        self._plot_venus_index(df, ax9)
        
        # 10. Prédictions futures
        ax10 = plt.subplot(5, 2, 10)
        self._plot_future_predictions(df, ax10)
        
        plt.suptitle(f'Analyse des Données Vénusiennes: {self.config["description"]} ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold', color='#FFD700')
        plt.tight_layout()
        plt.savefig(f'venus_{self.data_type}_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.show()
        
        # Générer les insights
        self._generate_venus_insights(df)
    
    def _plot_venus_cycle(self, df, ax):
        """Plot du cycle vénusien principal"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Valeur de base', 
               linewidth=2, color='#FFD700', alpha=0.9)
        
        ax.set_title(f'Cycle Vénusien Principal - {self.config["description"]}', 
                    fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel(self.config["unit"], color='#FFD700')
        ax.tick_params(axis='y', labelcolor='#FFD700')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # Ajouter des annotations pour les jours vénusiens
        for venus_day in range(0, 105, 10):  # Tous les 10 jours vénusiens
            earth_year = self.start_year + venus_day * 0.62
            if earth_year <= self.end_year:
                ax.axvline(x=earth_year, alpha=0.3, color='orange', linestyle='--')
                ax.text(earth_year, ax.get_ylim()[1]*0.9, f'J{venus_day}', 
                       rotation=90, color='orange', alpha=0.7, fontsize=8)
    
    def _plot_surface_conditions(self, df, ax):
        """Plot des conditions de surface vénusiennes"""
        ax.fill_between(df['Earth_Year'], df['Surface_Conditions'], alpha=0.7, 
                       color='#DAA520', label='Conditions de surface')
        
        ax.set_title('Conditions Extrêmes de Surface', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Facteur de conditions', color='#DAA520')
        ax.set_xlabel('Année Terrestre', color='white')
        ax.tick_params(axis='y', labelcolor='#DAA520')
        ax.tick_params(axis='x', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # Marquer les missions importantes
        missions = {
            1962: 'Mariner 2\n1er survol',
            1970: 'Venera 7\n1er atterrissage',
            1975: 'Venera 9/10\n1ères images',
            1982: 'Venera 13/14\nAtterrissages',
            1990: 'Magellan\nCartographie',
            2005: 'Venus Express',
            2010: 'Akatsuki'
        }
        
        for year, label in missions.items():
            if year in df['Earth_Year'].values:
                y_val = df.loc[df['Earth_Year'] == year, 'Surface_Conditions'].values[0]
                ax.annotate(label, xy=(year, y_val), xytext=(year, y_val*1.1),
                           arrowprops=dict(arrowstyle='->', color='yellow'),
                           color='yellow', fontsize=8, ha='center')
    
    def _plot_diurnal_variations(self, df, ax):
        """Plot des variations diurnes vénusiennes"""
        ax.plot(df['Earth_Year'], df['Diurnal_Variation'], label='Variation diurne', 
               color='#1E90FF', alpha=0.7)
        
        ax.set_title('Variations Diurnes (Très Faibles)', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Facteur de variation', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_atmospheric_effects(self, df, ax):
        """Plot des effets atmosphériques vénusiens"""
        ax.plot(df['Earth_Year'], df['Atmospheric_Effects'], label='Effet de serre', 
               linewidth=2, color='#FF4500')
        ax.plot(df['Earth_Year'], df['Volcanic_Influence'], label='Influence volcanique', 
               linewidth=2, color='#8B4513')
        
        ax.set_title('Effets Atmosphériques et Volcaniques', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Intensité relative', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_solar_day_phase(self, df, ax):
        """Plot de la phase du jour solaire"""
        scatter = ax.scatter(df['Earth_Year'], df['Solar_Day_Phase'], c=df['Solar_Day_Phase'], 
                           cmap='viridis', alpha=0.7, s=20)
        
        ax.set_title('Phase du Jour Solaire Vénusien (0-1)', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Phase du jour', color='white')
        ax.set_xlabel('Année Terrestre', color='white')
        plt.colorbar(scatter, ax=ax, label='Phase')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_smoothed_data_plot(self, df, ax):
        """Plot des données lissées"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Données brutes', 
               alpha=0.5, color='#FF6347')
        ax.plot(df['Earth_Year'], df['Smoothed_Value'], label='Données lissées (3 ans terrestres)', 
               linewidth=2, color='#00FF7F')
        
        ax.set_title('Données Brutes vs Lissées', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_hostility_level_plot(self, df, ax):
        """Plot du niveau d'hostilité environnementale"""
        ax.fill_between(df['Earth_Year'], df['Hostility_Level'], alpha=0.6, 
                       color='#FF4500', label='Niveau d\'hostilité')
        ax.plot(df['Earth_Year'], df['Hostility_Level'], color='#FF8C00', alpha=0.8)
        
        ax.set_title('Niveau d\'Hostilité Environnementale (0-100)', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Niveau d\'hostilité', color='white')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_cloud_variations(self, df, ax):
        """Plot des variations nuageuses"""
        ax.fill_between(df['Earth_Year'], df['Cloud_Variations'], alpha=0.6, 
                       color='#C0C0C0', label='Variations nuageuses')
        
        ax.set_title('Couverture Nuageuse et Variations', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Facteur de couverture', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_venus_index(self, df, ax):
        """Plot de l'indice vénusien composite"""
        ax.plot(df['Earth_Year'], df['Venus_Index'], label='Indice vénusien composite', 
               linewidth=2, color='#DA70D6')
        
        ax.set_title('Indice Vénusien Composite', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel('Valeur de l\'indice', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_future_predictions(self, df, ax):
        """Plot des prédictions futures"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Données historiques', 
               color='#FF6347', alpha=0.7)
        ax.plot(df['Earth_Year'], df['Future_Prediction'], label='Projections', 
               linewidth=2, color='#00FFFF', linestyle='--')
        
        ax.axvline(x=2020, color='yellow', linestyle=':', alpha=0.7, label='Début des prédictions')
        
        ax.set_title('Données Historiques et Projections Futures', fontsize=12, fontweight='bold', color='#FFD700')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _generate_venus_insights(self, df):
        """Génère des insights analytiques sur les données vénusiennes"""
        print(f"♀️ INSIGHTS ANALYTIQUES - {self.config['description']}")
        print("=" * 70)
        
        # 1. Statistiques de base
        print("\n1. 📊 STATISTIQUES FONDAMENTALES:")
        avg_value = df['Base_Value'].mean()
        max_value = df['Base_Value'].max()
        min_value = df['Base_Value'].min()
        current_value = df['Base_Value'].iloc[-1]
        
        print(f"Valeur moyenne: {avg_value:.2f} {self.config['unit']}")
        print(f"Valeur maximale: {max_value:.2f} {self.config['unit']}")
        print(f"Valeur minimale: {min_value:.2f} {self.config['unit']}")
        print(f"Valeur actuelle: {current_value:.2f} {self.config['unit']}")
        
        # 2. Analyse des caractéristiques uniques
        print("\n2. 🔄 CARACTÉRISTIQUES UNIQUES DE VÉNUS:")
        venus_day_current = df['Venus_Day'].iloc[-1]
        solar_day_duration = 0.62  # Années terrestres
        orbital_period = 0.62  # Années terrestres
        
        print(f"Jour vénusien actuel: {venus_day_current:.1f}")
        print(f"Durée du jour solaire: {solar_day_duration} années terrestres")
        print(f"Période orbitale: {orbital_period} années terrestres")
        print(f"Rotation rétrograde: Oui")
        print(f"Inclinaison axiale: 177.3° (presque inversée)")
        
        # 3. Conditions environnementales
        print("\n3. 🌡️ CONDITIONS ENVIRONNEMENTALES EXTRÊMES:")
        hostility_current = df['Hostility_Level'].iloc[-1]
        surface_conditions = df['Surface_Conditions'].iloc[-1]
        
        print(f"Niveau d'hostilité actuel: {hostility_current:.1f}%")
        print(f"Conditions de surface: {surface_conditions:.2f}x Terre")
        print(f"Température surface: ~462°C (constant)")
        print(f"Pression surface: ~92 bars (équivalent à 900m sous l'eau)")
        print(f"Atmosphère: 96.5% CO₂, nuages d'acide sulfurique")
        
        # 4. Événements majeurs
        print("\n4. 🚀 MISSIONS VÉNUSIENNES IMPORTANTES:")
        print("• 1962: Mariner 2 - premier survol réussi")
        print("• 1967-69: Venera 4,5,6 - premières entrées atmosphériques")
        print("• 1970: Venera 7 - premier atterrissage réussi")
        print("• 1975: Venera 9 et 10 - premières images de surface")
        print("• 1978: Pioneer Venus - étude atmosphérique")
        print("• 1982: Venera 13 et 14 - atterrissages avancés")
        print("• 1985: Vega 1 et 2 - ballons atmosphériques")
        print("• 1990: Magellan - cartographie radar complète")
        print("• 2005: Venus Express - étude atmosphérique")
        print("• 2010: Akatsuki - étude du climat")
        
        # 5. Phénomènes atmosphériques
        print("\n5. 💨 PHÉNOMÈNES ATMOSPHÉRIQUES UNIQUES:")
        print("• Super-rotation: vents à 300-400 km/h en haute atmosphère")
        print("• Effet de serre extrême: +500°C par rapport à sans atmosphère")
        print("• Nuages permanents: couverture complète d'acide sulfurique")
        print("• Double couche nuageuse: à 48-58 km et 50-70 km d'altitude")
        print("• Onde stationnaire: structure en forme de Y dans les nuages")
        
        # 6. Projections futures
        print("\n6. 🔮 PROJECTIONS ET MISSIONS FUTURES:")
        print("• 2029: Mission VERITAS de la NASA (planifiée)")
        print("• 2031: Mission DAVINCI+ de la NASA (planifiée)")
        print("• 2030s: Missions russes Venera-D")
        print("• Concepts avancés: dirigeables, stations flottantes")
        print("• Exploration humaine: extrêmement difficile mais étudiée")
        
        # 7. Implications scientifiques
        print("\n7. 🎯 IMPLICATIONS SCIENTIFIQUES:")
        if self.data_type == "temperature":
            print("• Compréhension des effets de serre extrêmes")
            print("• Modèle pour l'évolution climatique terrestre")
            print("• Étude des limites de l'habitabilité")
        
        elif self.data_type == "volcanic_activity":
            print("• Compréhension de l'activité géologique")
            print("• Comparaison avec le volcanisme terrestre")
            print("• Implications pour la jeunesse géologique")
        
        elif self.data_type == "wind_speeds":
            print("• Étude de la super-rotation atmosphérique")
            print("• Dynamique des fluides en conditions extrêmes")
            print("• Implications pour la météorologie planétaire")
        
        elif self.data_type == "atmospheric_composition":
            print("• Stabilité remarquable de la composition atmosphérique")
            print("• Compréhension des processus de dégazage")
            print("• Implications pour l'évolution planétaire")
        
        print("• Compréhension de l'évolution planétaire")
        print("• Recherche de la présence passée d'eau liquide")
        print("• Préparation pour l'exploration robotique avancée")

def main():
    """Fonction principale pour l'analyse des données vénusiennes"""
    # Types de données vénusiennes disponibles
    venus_data_types = [
        "temperature", "atmospheric_pressure", "cloud_cover", "surface_conditions",
        "volcanic_activity", "solar_radiation", "atmospheric_composition", "wind_speeds", "orbital_distance"
    ]
    
    print("♀️ ANALYSE DES DONNÉES NUMÉRIQUES DE VÉNUS (1960-2025)")
    print("=" * 65)
    
    # Demander à l'utilisateur de choisir un type de données
    print("Types de données vénusiennes disponibles:")
    for i, data_type in enumerate(venus_data_types, 1):
        analyzer_temp = VenusDataAnalyzer(data_type)
        print(f"{i}. {analyzer_temp.config['description']}")
    
    try:
        choix = int(input("\nChoisissez le numéro du type de données à analyser: "))
        if choix < 1 or choix > len(venus_data_types):
            raise ValueError
        selected_type = venus_data_types[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de la température par défaut.")
        selected_type = "temperature"
    
    # Initialiser l'analyseur
    analyzer = VenusDataAnalyzer(selected_type)
    
    # Générer les données
    venus_data = analyzer.generate_venus_data()
    
    # Sauvegarder les données
    output_file = f'venus_{selected_type}_data_1960_2025.csv'
    venus_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(venus_data[['Earth_Year', 'Venus_Day', 'Base_Value', 'Hostility_Level', 'Venus_Index']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse des données vénusiennes...")
    analyzer.create_venus_analysis(venus_data)
    
    print(f"\n✅ Analyse des données {analyzer.config['description']} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year} (années terrestres)")
    print(f"♀️ Couverture: ~{(2025-1960)/0.62:.1f} jours vénusiens")
    print("🌡️ Données: Conditions extrêmes, atmosphère, géologie")

if __name__ == "__main__":
    main()