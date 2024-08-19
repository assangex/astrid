# project astrid - rocket simulation 
# Written by Sabir Fisher, Aug 2024 

import pandas as pd
import numpy as np
from data.constants import GRAVITY, C_DRAG  

class RocketModel:
    def __init__(self, burn_time, m_dry, m_wet, thrust, area=.1016, time_set=0):
        self.burn_time = burn_time
        self.density = df['Density(kg/m^3)']
        self.time_set = time_set
        self.m_dry = m_dry
        self.m_wet = m_wet
        self.thrust = thrust
        self.area = area  # m^2
        self.initial_velocity = 0  # m/s
        self.initial_altitude = 0  # m
        self.time_increment = 0.1  # seconds
        self.results = []

        # Track maximum values
        self.max_mach = 0
        self.max_altitude = 0
        self.max_acceleration = 0
        self.max_g_force = 0
        self.max_q = 0

        # Calculate the propellant weight
        self.prop_weight = self.m_wet - self.m_dry

    def f_drag(self, v, area, density):
        return C_DRAG * density * (v ** 2) * 0.5 * area
    
    def weight(self, m):
        return m * GRAVITY

    def acceleration(self, f_net, m):
        return f_net / m

    def velocity(self, v0, a, t):
        return (a * t) + v0

    def altitude(self, h0, v0, t, a):
        return v0 * t + 0.5 * a * t**2 + h0
    
    def get_density(self, current_altitude):
        if pd.isna(current_altitude) or current_altitude < 0:
            return self.density.iloc[-1]  # Use the last known density as a fallback if altitude is NaN or negative

        rounded_altitude = int(current_altitude // 100)  # Round down to nearest 100 meters
        if rounded_altitude in self.density.index:
            density = self.density.loc[rounded_altitude]
        else:
            density = self.density.iloc[-1]  # Refer to previous density if altitude exceeds known range
        return density
    
    def mach(self, velocity, speed_of_sound=343):
        return velocity / speed_of_sound
    
    def maxq(self, density, velocity):
        return 0.5 * density * velocity

    def simulate(self):
        current_time = 0
        current_mass = self.m_wet  # Start with the wet mass (initial mass)
        current_altitude = self.initial_altitude  # Start from initial altitude

        # Initial conditions at t=0
        self.results.append({
            "Time": current_time,
            "Net Force": 0,
            "Drag Force": 0,
            "Thrust": 0,
            "Weight": self.weight(current_mass),
            "Acceleration": 0,
            "Velocity": 0,
            "Altitude": self.initial_altitude,
            "Density": self.get_density(self.initial_altitude),
            "Mach": 0,
            "Max Q": 0,
            "Status": "Climbing"
        })

        # Update initial conditions for the next step
        self.initial_velocity = 0
        self.initial_altitude = 0
        current_time += self.time_increment

        while current_time <= self.time_set:
            # Calculate mass based on the burn time
            if current_time < self.burn_time:
                current_mass -=  self.prop_weight / (self.burn_time / self.time_increment) # Mass calculation (I have no idea how burn time is being divided by the time increment but its working )
            else:
                current_mass = self.m_dry

            # Use the Thrust(N) value from the dataset for the current time step
            time_index = int(current_time // self.time_increment)
            if time_index < len(df):
                f_thrust = df['Thrust(N)'].iloc[time_index]
            else:
                f_thrust = 0  # No thrust after the provided thrust data ends

            # Grab the current density based on altitude
            density = self.get_density(current_altitude)

            # Calculate net force, acceleration, velocity, altitude, drag force, max q
            f_drag = self.f_drag(self.initial_velocity, self.area, density)
            f_net = f_thrust - f_drag - self.weight(current_mass)
            a = self.acceleration(f_net, current_mass)
            v = self.velocity(self.initial_velocity, a, self.time_increment)
            mach_value = self.mach(v)
            h = self.altitude(self.initial_altitude, self.initial_velocity, self.time_increment, a)
            maxq = self.maxq(density, v)
            status = "Climbing" if h > self.initial_altitude else "Descending"
            if status == "Descending" and h < 0:
                break

            # Track maximum values
            if mach_value > self.max_mach:
                self.max_mach = mach_value
            if h > self.max_altitude:
                self.max_altitude = h
            if a > self.max_acceleration:
                self.max_acceleration = a
            g_force = a / GRAVITY
            if g_force > self.max_g_force:
                self.max_g_force = g_force
            if maxq > self.max_q:
                self.max_q = maxq

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

            # Update initial conditions for the next step
            self.initial_velocity = v
            self.initial_altitude = h
            current_altitude = h  # Update current altitude
            current_time += self.time_increment

        return self.results

    def print_results(self):
        for result in self.results:
            print(f"Time: {result['Time']:.2f} s, Force: {result['Net Force']:.2f} N, Density: {result['Density']:.3f} kg/m³, "
                  f"Drag Force: {result['Drag Force']:.2f} N, Weight: {result['Weight']:.2f} N, "
                  f"Acceleration: {result['Acceleration']:.2f} m/s², Velocity: {result['Velocity']:.2f} m/s (Mach {result['Mach']:.2f}), "
                  f"Altitude: {result['Altitude']:.2f} m, Status: {result['Status']}")
            

        print(f"\nMax Mach: {self.max_mach:.2f}")
        print(f"Apogee: {self.max_altitude:.2f} m")
        print(f"Max Acceleration: {self.max_acceleration:.2f} m/s²")
        print(f"Max G's: {self.max_g_force:.2f} g")
        print(f"Max Q: {self.max_q:.2f} Pa")

if __name__ == "__main__":
    # Load the data
    df = pd.read_csv('/home/sabir/Apps/Code/code/projects/astrid/data/data.csv')
    df = df.fillna(0)
    
    # Extract the burn time from the 'BurnTime(s)' column
    burn_time = 7 #df['BurnTime(s)'].tail(1).iloc[-1]  # Taking the last row's burn time for the simulation
    thrust = df["Thrust(N)"].head(1).iloc[0]  # Taking the first row's thrust for the simulation
    
    # Set other parameters
    m_dry = 1.4500  # in kg
    m_wet = 5.4500  # in kg
    time_set = 14  # Total simulation time in seconds

    # Create RocketModel instance
    rocket = RocketModel(burn_time=burn_time, thrust=thrust, m_dry=m_dry, m_wet=m_wet, time_set=time_set)

    # Run simulation
    rocket.simulate()

    # Print results
    rocket.print_results()
