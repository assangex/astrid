import pandas as pd
import numpy as np
from data.constants import GRAVITY, C_DRAG
import logging
import os 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

class RocketModel:
    def __init__(self, df: pd.DataFrame, burn_time: float, m_dry: float, m_wet: float, thrust: float, of_ratio: float, prop_frac: float, area: float = 0.1016, time_inc: float = 0.1):
        self.burn_time = burn_time
        self.of_ratio = of_ratio
        self.prop_frac = prop_frac
        self.density = df['Density(kg/m^3)']
        self.m_dry = m_dry
        self.m_wet = m_wet * (1 - prop_frac) + m_dry  
        self.thrust = thrust
        self.area = area  # m^2    
        self.i_velocity = 0  # m/s
        self.i_altitude = 0  # m
        self.time_inc = time_inc 
        self.results = []

        # Calculate the propellant weight
        self.prop_weight = self.m_wet - self.m_dry
        self.fuel_mass = self.prop_weight / (1 + self.of_ratio)
        self.ox_mass = self.prop_weight - self.fuel_mass

        # Track maximum values
        self.max_mach = 0
        self.max_altitude = 0
        self.max_acceleration = 0
        self.max_g_force = 0
        self.max_q = 0

    def f_drag(self, v: float, area: float, density: float) -> float:
        return C_DRAG * density * (v ** 2) * 0.5 * area

    def weight(self, m: float) -> float:
        return m * GRAVITY

    def acceleration(self, f_net: float, m: float) -> float:
        return f_net / m

    def velocity(self, v0: float, a: float, t: float) -> float:
        return (a * t) + v0

    def altitude(self, h0: float, v0: float, t: float, a: float) -> float:
        return v0 * t + 0.5 * a * t**2 + h0

    def get_density(self, current_altitude: float) -> float:
        if pd.isna(current_altitude) or current_altitude < 0:
            return self.density.iloc[-1]  # Use the last known density as a fallback if altitude is NaN or negative

        rounded_altitude = int(current_altitude // 100)  # Round down to nearest 100 meters
        if rounded_altitude in self.density.index:
            density = self.density.loc[rounded_altitude]
        else:
            density = self.density.iloc[-1]  # Refer to previous density if altitude exceeds known range
        return density

    def mach(self, velocity: float, speed_of_sound: float = 343) -> float:
        return velocity / speed_of_sound

    def maxq(self, density: float, velocity: float) -> float:
        return 0.5 * density * velocity

    def deployment(self, altitude: float, velocity: float, main_dep: float, drogue_dep: float, density: float, main_cd: float, drogue_cd: float, drag_coefficient: float) -> float:
        if velocity < 0:
            if altitude < main_dep:
                return -0.5 * main_cd * density * velocity**2 * self.area
            else:
                return -0.5 * drogue_cd * density * velocity**2 * self.area
        else:
            return 0.5 * drag_coefficient * density * velocity**2 * self.area

    def simulate(self) -> list:
        current_time = 0
        current_mass = self.m_wet  
        current_altitude = self.i_altitude  

        # i conditions at t=0
        self.results.append({
            "Time": current_time,
            "Net Force": 0,
            "Drag Force": 0,
            "Thrust": 0,
            "Weight": self.weight(current_mass),
            "Acceleration": 0,
            "Velocity": 0,
            "Altitude": self.i_altitude,
            "Density": self.get_density(self.i_altitude),
            "Mach": 0,
            "Max Q": 0,
            "Status": "Climbing"
        })

        # Update i conditions for the next step
        self.i_velocity = 0
        self.i_altitude = 0
        current_time += self.time_inc

        while True:
            # Calculate mass based on the burn time
            if current_time < self.burn_time:
                fuel_consumed = (self.fuel_mass / self.burn_time) * self.time_inc
                ox_consumed = (self.ox_mass / self.burn_time) * self.time_inc
                current_mass -= (fuel_consumed + ox_consumed)
            else:
                current_mass = self.m_dry

            # Use the Thrust(N) value from the dataset for the current time step
            time_index = int(current_time // self.time_inc)
            if time_index < len(self.density):
                f_thrust = self.thrust
            else:
                f_thrust = 0  # No thrust after the provided thrust data ends

            # Grab the current density based on altitude
            density = self.get_density(current_altitude)

            # Calculate net force, acceleration, velocity, altitude, drag force, max q
            f_drag = self.deployment(current_altitude, self.i_velocity, 300, 3049.753836, density, 1.2, 0.97, C_DRAG)
            f_net = f_thrust - f_drag - self.weight(current_mass)
            a = self.acceleration(f_net, current_mass)
            
            
            v = self.velocity(self.i_velocity, a, self.time_inc)
            mach_value = self.mach(v)
            h = self.altitude(self.i_altitude, self.i_velocity, self.time_inc, a)
            maxq = self.maxq(density, v)
            status = "Climbing" if h > self.i_altitude else "Descending"
            if status == "Descending" and h < 0:
                 break

            # Track maximum values
            self.max_mach = max(self.max_mach, mach_value)
            self.max_altitude = max(self.max_altitude, h)
            self.max_acceleration = max(self.max_acceleration, a)
            g_force = a / GRAVITY
            self.max_g_force = max(self.max_g_force, g_force)
            self.max_q = max(self.max_q, maxq)

            # Store the results for the current time step
            self.results.append({
                "Time": current_time,
                "Net Force": f_net,
                "Drag Force": f_drag,
                "Thrust": f_thrust,
                "Weight": self.weight(current_mass),
                "Acceleration": a,
                "Velocity": v,
                "Altitude": h,
                "Density": density,
                "Mach": mach_value,
                "Status": status
            })

            # Update i conditions for the next step
            self.i_velocity = v
            self.i_altitude = h
            current_altitude = h  # Update current altitude
            current_time += self.time_inc

        return self.results

    def save_results(self, filename: str):
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filename, index=False)
        logging.info(f"Results saved to {filename}")
        if os.path.exists(filename):
            logging.info(f"File size: {os.path.getsize(filename) / 1024:.2f}")

    def print_results(self):
        for result in self.results:
            logging.info(f"Time: {result['Time']:.2f} s, Force: {result['Net Force']:.2f} N, Density: {result['Density']:.3f} kg/m³, "
                         f"Drag Force: {result['Drag Force']:.2f} N, Weight: {result['Weight']:.2f} N, "
                         f"Acceleration: {result['Acceleration']:.2f} m/s², Velocity: {result['Velocity']:.2f} m/s (Mach {result['Mach']:.2f}), "
                         f"Altitude: {result['Altitude']:.2f} m, Status: {result['Status']}")

        logging.info(f"\nMax Mach: {self.max_mach:.2f}")
        logging.info(f"Apogee: {self.max_altitude:.2f} m")
        logging.info(f"Max Acceleration: {self.max_acceleration:.2f} m/s²")
        logging.info(f"Max G's: {self.max_g_force:.2f} g")
        logging.info(f"Max Q: {self.max_q:.2f} Pa")


if __name__ == "__main__":
    try:
        # Load the data
        df = pd.read_csv('/home/sabir/Apps/Code/code/projects/astrid/data/data.csv')
        df = df.fillna(0)

        #'BurnTime(s)' column
        burn_time = 13.5 
        thrust = 4500

        # Set other parameters
        m_dry = 1.4500  # in kg
        m_wet = 100  # in kg
        of_ratio = 2.5  # Filler o/f
        prop_frac = 0.9  # Filler propfrac

        # RocketModel instance
        rocket = RocketModel(df, burn_time=burn_time, thrust=thrust, m_dry=m_dry, m_wet=m_wet, of_ratio=of_ratio, prop_frac=prop_frac)
        # Run simulation
        rocket.simulate()
        # result saving 
        rocket.save_results('astrid.csv')

        # Print result        rocket.print_results()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
