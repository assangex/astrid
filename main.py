#Made by assangex <3 
#v.1.9
import pandas as pd
import numpy as np
import logging
import os
import math
import matplotlib.pyplot as plt
from data.constants import GRAVITY
from libs.logo import astrid_logo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def clamp(value, min_val, max_val):
    return max(min(value, max_val), min_val)


class Astrid:
    def __init__(self, df: pd.DataFrame, thrust_data: pd.Series, burn_time: float, m_wet: float, of_ratio: float, prop_frac: float, area: float, oxidizer_density: float, ox_pressure: float, internal_radius: float, tank_thickness: float, material_yield_strength: float, safety_factor: float, time_inc: float = 0.10, launch_altitude: float = 0, cd_data: pd.DataFrame = None):
        # Existing parameters
        self.burn_time = burn_time
        self.of_ratio = of_ratio
        self.prop_frac = prop_frac
        self.density = df["Density(kg/m^3)"].tolist()
        self.m_wet = m_wet
        self.thrust_data = thrust_data
        self.area = area
        self.time_inc = time_inc
        self.i_altitude = launch_altitude
        self.cd_data = cd_data
        self.max_altitude = max
        self.results = []

        # Oxidizer Tank Parameters
        self.oxidizer_density = oxidizer_density
        self.ox_pressure = ox_pressure
        self.internal_radius = internal_radius
        self.tank_thickness = tank_thickness
        self.material_yield_strength = material_yield_strength
        self.safety_factor = safety_factor

        # Calculate initial oxidizer properties
        self.total_ox_mass = self.prop_frac * self.m_wet  # Total oxidizer mass
        self.oxidizer_mass_flow_rate = self.total_ox_mass / burn_time
        self.tank_volume = self.total_ox_mass / self.oxidizer_density  # in cubic meters

        # Initialize mass properties
        self.total_prop_mass = self.prop_frac * m_wet  # Total propellant mass
        self.fuel_mass = self.total_prop_mass / (1 + self.of_ratio)  # Fuel mass based on O/F ratio
        self.ox_mass = self.total_prop_mass - self.fuel_mass  # Oxidizer mass
        
        self.current_mass = m_wet  # Initial wet mass of the rocket

        # Stress calculations
        self.hoop_stress = (self.ox_pressure * self.internal_radius) / self.tank_thickness
        self.longitudinal_stress = (self.ox_pressure * self.internal_radius) / (2 * self.tank_thickness)

        # factor of safety checker
        self.actual_safety_factor = self.material_yield_strength / max(self.hoop_stress, self.longitudinal_stress)
        if self.actual_safety_factor < self.safety_factor:
            logging.warning(f"Warning: Actual safety factor {self.actual_safety_factor:.2f} is below the required safety factor of {self.safety_factor}")

        # Log results
        logging.info(f"Oxidizer Mass Flow Rate: {self.oxidizer_mass_flow_rate:.2f} kg/s")
        logging.info(f"Hoop Stress: {self.hoop_stress:.2f} Pa")
        logging.info(f"Longitudinal Stress: {self.longitudinal_stress:.2f} Pa")
        logging.info(f"Tank Volume: {self.tank_volume:.2f} m^3")
        logging.info(f"Safety Factor: {self.actual_safety_factor:.2f}") 

        # Existing initialization logic...
        self.mass_dot = self.total_prop_mass / burn_time
        self.isp = self.thrust_data.mean() / (self.mass_dot * GRAVITY)
        self.total_impulse = self.thrust_data.sum() * self.time_inc
        self.average_thrust = self.thrust_data.mean()

        self.main_cd = 1.9
        self.main_diameter = 3.6576
        self.drogue_cd = 0.97
        self.drogue_diameter = 0.9144
        self.main_area = math.pi * (self.main_diameter / 2) ** 2
        self.drogue_area = math.pi * (self.drogue_diameter / 2) ** 2

        logging.info(f"Specific Impulse (Isp): {self.isp:.2f} seconds")
        logging.info(f"Mass Flow Rate: {self.mass_dot:.4f} kg/s")
        logging.info(f"Total Impulse: {self.total_impulse:.2f} Ns")
        logging.info(f"Average Thrust: {self.average_thrust:.2f} N")

    def get_cd(self, time: float, altitude: float, mach: float) -> float:
        if time < self.burn_time:
            cd = np.interp(burn_time, self.cd_data["Cd power on"], self.cd_data["CD"])
        else:
            cd = np.interp(time, self.cd_data["Cd power off"], self.cd_data["CD"])
            # After burn time, determine the Cd based on coast and altitude
            # if altitude > 1250:
            #     cd = self.drogue_cd
            # else:
            #     cd = self.main_cd

        return cd

    def drag(self, altitude: float, velocity: float, density: float, time: float) -> float:
        mach_value = self.mach(velocity)
        cd = self.get_cd(time, altitude, mach_value)
        if velocity < 0:
            if altitude <= 1000:
                return -0.5 * self.main_cd * density * velocity**2 * self.main_area
            elif altitude <= 1375:
                return -0.5 * self.drogue_cd * density * velocity**2 * self.drogue_area
        return 0.5 * cd * density * velocity**2 * self.area

    def weight(self) -> float:
        return self.current_mass * GRAVITY

    def acceleration(self, force: float) -> float:
        return force / self.current_mass

    def velocity(self, v0: float, a: float) -> float:
        return v0 + a * self.time_inc

    def altitude(self, h0: float, v0: float, a: float) -> float:
        return h0 + v0 * self.time_inc + 0.5 * a * self.time_inc**2

    def get_density(self, altitude: float) -> float:
        index = clamp(int(altitude // 50), 0, len(self.density) - 1)
        return self.density[index]

    def ve(self) -> float:
        return self.isp * GRAVITY

    def f_thrust(self, current_time: float) -> float:
        index = clamp(int(current_time // self.time_inc), 0, len(self.thrust_data) - 1)
        return self.thrust_data.iloc[index]

    def mach(self, velocity: float, speed_of_sound: float = 343) -> float:
        return velocity / speed_of_sound

    def maxq(self, density: float, velocity: float) -> float:
        return 0.5 * density * velocity**2

    def simulate(self):
        current_time = 0
        current_velocity = 0
        current_altitude = self.i_altitude
        main_deployed = False
        drogue_deployed = False

        self.results.append({
            "Time": current_time,
            "Net Force": 0,
            "Drag Force": 0,
            "Thrust": 0,
            "Weight": self.weight(),
            "Acceleration": 0,
            "Velocity": current_velocity,
            "Altitude": current_altitude,
            "Density": self.get_density(current_altitude),
            "Mach": 0,
            "Max Q": 0,
            "Status": "Climbing"
        })

        while True:
            if current_time < self.burn_time:
                fuel_consumed = (self.fuel_mass / self.burn_time) * self.time_inc
                ox_consumed = (self.ox_mass / self.burn_time) * self.time_inc
                self.current_mass -= (fuel_consumed + ox_consumed)
            else:
                self.current_mass = self.m_wet - self.total_prop_mass

            density = self.get_density(current_altitude)
            f_drag = self.drag(current_altitude, current_velocity, density, current_time)
            f_net = self.f_thrust(current_time) - f_drag - self.weight()
            a = self.acceleration(f_net)
            current_velocity = self.velocity(current_velocity, a)
            mach_value = self.mach(current_velocity)
            current_altitude = self.altitude(current_altitude, current_velocity, a)
            maxq = self.maxq(density, current_velocity)
            status = "Climbing" if current_altitude > self.i_altitude else "Descending"

            if not main_deployed and current_altitude <= 1000 and current_velocity < 0:
                status = "Main Parachute Deployed"
                main_deployed = True

            if not drogue_deployed and current_altitude <= 1375 and current_velocity < 0:
                status = "Drogue Parachute Deployed"
                drogue_deployed = True

            if current_altitude > self.max_altitude:
                self.max_altitude = current_altitude
                self.apogee_time = current_time

            self.max_mach = max(self.max_mach, mach_value)
            self.max_acceleration = max(self.max_acceleration, a)
            self.max_velocity = max(self.max_velocity, current_velocity)
            g_force = a / GRAVITY
            self.max_g_force = max(self.max_g_force, g_force)
            self.max_q = max(self.max_q, maxq)

            self.results.append({
                "Time": current_time,
                "Net Force": f_net,
                "Drag Force": f_drag,
                "Thrust": self.f_thrust(current_time),
                "Weight": self.weight(),
                "Acceleration": a,
                "Velocity": current_velocity,
                "Altitude": current_altitude,
                "Density": density,
                "Mach": mach_value,
                "Status": status
            })

            current_time += self.time_inc

            if status == "Descending" and current_altitude < 0:
                break

        self.m_dry = self.current_mass
        logging.info(f"Final Dry Mass: {self.m_dry:.2f} kg")
        logging.info(f"Time of Apogee: {self.apogee_time:.2f} seconds")

    def save(self, filename: str):
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")
        if os.path.exists(filename):
            logging.info(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

    def plot(self):
        times = [result['Time'] for result in self.results]
        altitudes = [result['Altitude'] for result in self.results]
        thrusts = [result["Thrust"] for result in self.results]
        velocities = [result['Velocity'] for result in self.results]
        accelerations = [result['Acceleration'] for result in self.results]

        plt.figure(figsize=(10, 8))

        plt.subplot(3, 1, 1)
        plt.plot(times, altitudes, label='Altitude (m)', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Altitude (m)')
        plt.title('Rocket Altitude over Time')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(times, velocities, label='Velocity (m/s)', color='green')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (m/s)')
        plt.title('Rocket Velocity over Time')
        plt.grid(True)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(times, thrusts, label='Thrust (N)', color='purple')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²) / Thrust (N)')
        plt.title('Thrust over Time')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

def print_results(self):
    # Loop through and print each result from the simulation
    for result in self.results:
        logging.info(f"Time: {result['Time']:.2f} s, "
                     f"Net Force: {result['Net Force']:.2f} N, "
                     f"Density: {result['Density']:.3f} kg/m³, "
                     f"Drag Force: {result['Drag Force']:.2f} N, "
                     f"Weight: {result['Weight']:.2f} N, "
                     f"Acceleration: {result['Acceleration']:.2f} m/s², "
                     f"Velocity: {result['Velocity']:.2f} m/s (Mach {result['Mach']:.2f}), "
                     f"Altitude: {result['Altitude']:.2f} m, "
                     f"Status: {result['Status']}")

    # Print out overall results and key performance metrics
    astrid_logo()  # Logo display
    
    logging.info(f"\n===== ROCKET PERFORMANCE RESULTS =====")
    logging.info(f"Max Mach: {self.max_mach:.2f}")
    logging.info(f"Apogee: {self.max_altitude:.2f} meters at {self.apogee_time:.2f} seconds")
    logging.info(f"Max Acceleration: {self.max_acceleration:.2f} m/s²")
    logging.info(f"Max G's: {self.max_g_force:.2f} g")
    logging.info(f"Max Q: {self.max_q:.2f} Pa")
    logging.info(f"Max Velocity: {self.max_velocity:.2f} m/s")
    
    logging.info(f"\n===== MASS AND PROPULSION STATS =====")
    logging.info(f"Initial Wet Mass: {self.m_wet:.2f} kg")
    logging.info(f"Final Dry Mass: {self.m_dry:.2f} kg")
    logging.info(f"Oxidizer Mass: {self.ox_mass:.2f} kg")
    logging.info(f"Fuel Mass: {self.fuel_mass:.2f} kg")
    logging.info(f"Exhaust Velocity (Ve): {self.ve():.2f} m/s")
    logging.info(f"Specific Impulse (Isp): {self.isp:.2f} seconds")
    logging.info(f"Total Impulse: {self.total_impulse:.2f} Ns")
    logging.info(f"Mass Flow Rate: {self.mass_dot:.4f} kg/s")
    
    logging.info(f"\n===== OXIDIZER TANK DETAILS =====")
    logging.info(f"Oxidizer Density: {self.oxidizer_density:.2f} kg/m³")
    logging.info(f"Oxidizer Tank Pressure: {self.ox_pressure:.2f} Pa")
    logging.info(f"Oxidizer Mass Flow Rate: {self.oxidizer_mass_flow_rate:.2f} kg/s")
    logging.info(f"Oxidizer Tank Volume: {self.tank_volume:.2f} m³")
    logging.info(f"Hoop Stress: {self.hoop_stress:.2f} Pa")
    logging.info(f"Longitudinal Stress: {self.longitudinal_stress:.2f} Pa")
    logging.info(f"Material Yield Strength: {self.material_yield_strength:.2f} Pa")
    logging.info(f"Actual Safety Factor: {self.actual_safety_factor:.2f}")
    logging.info(f"Required Safety Factor: {self.safety_factor:.2f}")

    logging.info(f"=======================================")


def adjust_data_length(data: pd.Series, target_length: int) -> pd.Series:
    actual_length = len(data)
    
    if actual_length < target_length:
        # If the data points are fewer, interpolate to match the points
        data = np.interp(
            np.linspace(0, actual_length - 1, target_length),
            np.arange(actual_length),
            data
        )
    elif actual_length > target_length:
        # If the data points are more, truncate to match the points
        data = data.iloc[:target_length]

    return pd.Series(data)

if __name__ == "__main__":
    try:
        df = pd.read_csv('/home/sabir/Apps/learning/code/code/projects/astrid/data/data.csv').fillna(0)
        cd_data = pd.read_csv('/home/sabir/Apps/learning/code/code/projects/astrid/data/variablecd.csv')
        thrust_df = pd.read_csv('/home/sabir/Apps/learning/code/code/projects/astrid/data/thrustcurve.csv')
        thrust_df['New Thrust'] = thrust_df['Thrust'] * 4

        burn_time = 12
        thrust = 5500
        m_wet = 65
        of_ratio = 5.5
        prop_frac = 0.53
        area = math.pi * (8 * 0.0254 / 2) ** 2
        launch_altitude = 1350
        time_inc = 0.10

        # New oxidizer tank parameters
        oxidizer_density = 452  # Example: Density of liquid N2O in kg/m^3
        ox_pressure = 2000 * 6894.76  # Convert psi to Pascals (1 psi = 6894.76 Pa)
        internal_radius = 0.127  # Example: 5 inches converted to meters (1 inch = 0.0254 meters)
        tank_thickness = 0.01  # Example: 10 mm wall thickness (can be changed based on materials)
        material_yield_strength = 276e6  # Example: 6061-T6 Aluminum yield strength in Pascals
        safety_factor = 2

        # Adjust the thrust data length to match the required number of points
        required_points = int(burn_time / time_inc)
        adjusted_thrust_data = adjust_data_length(thrust_df['New Thrust'], required_points)

        # Create the rocket simulation object with all required parameters
        rocket = Astrid(df, 
                        burn_time=burn_time, 
                        m_wet=m_wet, 
                        thrust_data=adjusted_thrust_data, 
                        of_ratio=of_ratio, 
                        prop_frac=prop_frac, 
                        launch_altitude=launch_altitude, 
                        cd_data=cd_data, 
                        area=area, 
                        time_inc=time_inc,
                        oxidizer_density=oxidizer_density, 
                        ox_pressure=ox_pressure, 
                        internal_radius=internal_radius, 
                        tank_thickness=tank_thickness, 
                        material_yield_strength=material_yield_strength, 
                        safety_factor=safety_factor)

        rocket.simulate()
        rocket.save('astrid.csv')
        rocket.print_results()
        rocket.plot()

        

    except Exception as e:
        logging.error(f"An error occurred: {e}")
