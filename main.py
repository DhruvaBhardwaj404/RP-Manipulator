import math
import numpy as np
import pandas
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import odeint, solve_ivp
from scipy.integrate import quad



class DHParam:

    def __init__(self, b, a, theta, alpha, m):
        self.b = b
        self.a = a
        self.theta = theta
        self.alpha = alpha
        self.m = m


def calculate_t1_pr(theta1, a1):
    theta1 = math.radians(theta1)
    return np.array([[round(np.cos(theta1), 5), 0, round(np.sin(theta1), 5), round(a1*np.cos(theta1), 5)],
                     [round(np.sin(theta1), 5), 0, -round(np.cos(theta1), 5), round(a1*np.sin(theta1), 5)],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]
                    ])


def calculate_t2_pr(b2):
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, b2],
                     [0, 0, 0, 1]])


def calculate_t_pr(theta1, b2, a1):
    #print(theta1)
    theta1 = math.radians(theta1)
    #print(theta1)
    return np.array([[round(math.cos(theta1), 3), round(-math.sin(theta1), 3), 0, round(a1*math.cos(theta1) - b2*math.sin(theta1), 3)],
                    [round(math.sin(theta1), 3), round(math.cos(theta1), 3), 0, round(a1*math.sin(theta1) + b2*math.cos(theta1), 3)],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]])
class RKDC:
    """

    """
    def __init__(self, l1, l2):
        self.l1 = l1
        self.l2 = l2
        self.fd_b = []
        self.fd_theta = []
        self.fd_theta_dt2 = []
        self.fd_b_dt2 = []
        self.fd_f = []
        self.fd_t = []
        self.fd_x2 = []
        self.fd_y2 = []
        self.diff_t = []
        self.fd_theta_dt =[]
        self.fd_b_dt = []
        self.t = 0

    def forward_kinematics(self, theta1_end, b2_end, theta1_start=None, b2_start=None):
        if theta1_start is None:
            theta1_start = self.l1.theta
        if b2_start is None:
            b2_start = self.l2.b

        T1_start = calculate_t1_pr(theta1_start, self.l1.a)
        T2_start = calculate_t2_pr(b2_start)
        T_start =  np.matmul(T1_start, T2_start).round(3)
        #T_start = calculate_t_pr(theta1_start, b2_start, self.l1.a)
        print("T initial\nT1:\n", T1_start)
        print("\nT2:\n", T2_start)
        print("\nT :\n", T_start, "\n")

        T1_end = calculate_t1_pr(theta1_end, self.l1.a)
        T2_end = calculate_t2_pr(b2_end)
        T_end = np.matmul(T1_end, T2_end).round(3)
        #T_end = calculate_t_pr(theta1_end, b2_end, self.l1.a)
        print("T End\nT1:\n", T1_end)
        print("\nT2:\n", T2_end)
        print("\nT :\n", T_end)
        T = 1
        data = []
        slope_x = T_end[0][3] - T_start[0][3]
        slope_y = T_end[1][3] - T_start[1][3]
        t = 0
        while t <= T:
            arg = (6.18 / T) * t
            x = T_start[0][3] + slope_x*(t - (T/6.18)*np.sin(arg))
            y = T_start[1][3] + slope_y*(t - (T/6.18)*np.sin(arg))
            data.append([x, y, t])
            t += 1/100

        df = pd.DataFrame(data, columns=["x", "y", "timestep"])
        print(df)
        df.plot.line(x="timestep", y="x")
        df.plot.line(x="timestep", y="y")


        plt.show()

    def inverse_kinematics(self, px, py):
        A = self.l1.a + px
        B = -2*py
        C = self.l1.a - px

        z1 = (-B + np.sqrt(B*B - 4*A*C))/(2*A)
        z2 = (-B - np.sqrt(B*B - 4*A*C))/(2*A)

        print(f"Inverse Dynamics Solutions for ({px, py}) =>\n")
        theta1 = 2 * math.atan(z1)
        b2_1 = round(px * np.sin(theta1) - py * np.cos(theta1), 2)
        print("solution 1=> ", math.degrees(theta1), b2_1)

        theta2 = 2 * math.atan(z2)
        b2_2 = round(px * np.sin(theta2) - py * np.cos(theta2),2)
        print("solution 2=> ", math.degrees(theta2), b2_2)

    def inverse_dynamics(self, theta_end, b_end, T):
        t = 0

        slope_theta = (theta_end - self.l1.theta)/T
        slope_theta_rad = np.radians(slope_theta)
        slope_b = b_end - self.l2.b
        temp = []

        while t <= T:
            arg = (6.18 / T) * t
            theta = self.l1.theta + slope_theta*(t - (T/6.18)*np.sin(arg))
            theta_dt = slope_theta * (1 - np.cos(arg))
            theta_dt2 = slope_theta * ((6.14 / T) * np.sin(arg))

            theta_rad = np.radians(theta)
            theta_rad_dt = slope_theta_rad*(1-np.cos(arg))
            theta_rad_dt2 = slope_theta_rad*((6.14/T) * np.sin(arg))


            b = self.l2.b + slope_b*(t - (T/6.18)*np.sin(arg))
            b_dt = slope_b * (1-np.cos(arg))
            b_dt2 = slope_b * ((6.14/T) * np.sin(arg))

            torque = ((self.l1.m * (self.l1.a**2))/3 + self.l2.m*(self.l1.a**2) + self.l2.m*(b**2))*theta_rad_dt2 + (2 * b * b_dt * theta_rad_dt) * self.l2.m - self.l2.m*self.l1.a*b_dt2
            force = (-self.l2.m*self.l1.a)*theta_rad_dt2 + (self.l2.m)*b_dt2 - self.l2.m*b*theta_rad_dt**2

            temp.append([theta, theta_dt, theta_dt2, b, b_dt, b_dt2, torque, force, t])
            t += T/100
        data = pandas.DataFrame(data=temp,
                                columns=["theta", "theta_dt", "theta_dt2", "b", "b_dt", "b_dt2", "torque", "force", "timestep"])
        self.fd_t = data["torque"].to_numpy()
        self.fd_f = data["force"].to_numpy()
        fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(5, 5))
        i = 0

        for col in ["theta", "theta_dt", "theta_dt2", "b", "b_dt", "b_dt2", "torque", "force"]:

            sns.lineplot(data, x="timestep", y=col, ax=ax[i//3][i % 3])
            i += 1

        print(data[["torque", "force"]])
        plt.show()

    def fd_rev_joint_acceleration(self, ti, y):

        x1, x2, y1, y2 = y
        self.diff_t.append(ti)

        t = self.t % 100
        self.t += 1

        if t > 0:

            theta_dt = (1/((self.l1.m*self.l1.a**2)/3 + self.l2.m*self.l1.a**2 + self.l2.m*self.fd_b[-1])) * (self.fd_t[-1] + self.l1.a * self.fd_b[-1])
            b_dt = y1 + self.l1.a * x1 + self.fd_theta[t-1]*self.fd_b[t-1]

            self.fd_theta_dt.append(np.degrees(theta_dt))
            self.fd_b_dt.append(b_dt)
            self.fd_theta.append(self.fd_theta[-1] + theta_dt)
            self.fd_b.append(self.fd_b[-1] + b_dt)

        b = self.fd_b[-1]

        tor = self.fd_t[t]
        f = self.fd_f[t]

        cf = (tor + self.l1.a*self.fd_b_dt2[-1] - 2*self.fd_b[-1]*self.fd_b_dt[-1]*self.fd_theta_dt[-1]*self.l2.m)/((self.l1.m*self.l1.a**2)/3 + self.l2.m*self.l1.a**2 + self.l2.m*self.fd_b[-1])

        if(t>0):
            self.fd_theta_dt2.append(cf)

        df = f + self.l1.a*self.fd_theta_dt[-1] + b*self.fd_theta[-1]**2

        if(t>0):
            self.fd_b_dt2.append(df)

        return [x2,
                cf,
                self.fd_b_dt[-1],
                y2,
                ]

    def forward_dynamics(self, theta_start, b_start, theta_dt_start=0.0, theta_dt2_start=0.0, b_dt_start=0.0, b_dt2_start=0.0 ):

        t = [x/100 for x in range(0, 100, 1)]

        self.fd_b.append(b_start)
        self.fd_theta.append(theta_start)
        self.fd_b_dt.append(b_dt_start)
        self.fd_theta_dt.append(theta_dt_start)
        self.fd_theta_dt2.append(theta_dt2_start)
        self.fd_b_dt2.append(b_dt2_start)

        y = (theta_start, self.fd_t[0], b_start, self.fd_b[0])

        data = solve_ivp(self.fd_rev_joint_acceleration, (0, 1), y)

        time = [x for x in range(0,len(self.fd_theta_dt2))]

        for i in range(0, 100):
            self.fd_theta = np.degrees(self.fd_theta)

        data = pd.DataFrame({"theta":self.fd_theta[0:100], "theta_dt":self.fd_theta_dt[0:100],"b":self.fd_b[0:100],"b_dt":self.fd_b_dt[0:100],"theta_dt2":self.fd_theta_dt2[0:100],"b_dt2":self.fd_b_dt2[0:100],"t":time[0:100]})
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(5, 5))
        i = 0

        for col in ["theta", "theta_dt", "theta_dt2", "b", "b_dt", "b_dt2"]:
            sns.lineplot(data, x="t", y=col, ax=ax[i // 3][i % 3])
            i += 1


        plt.show()


if __name__ == '__main__':
   link1 = DHParam(0, 0.1, 0, 90, 1)
   link2 = DHParam(0, 0, 0, 0, 1)

   RP_man = RKDC(link1, link2)
   #RP_man.forward_kinematics(135, 0.1, 0, 0)
   print("\n\n")
   #RP_man.inverse_kinematics(0, 0.141)
   RP_man.inverse_dynamics(90, 0.1, 1)
   RP_man.forward_dynamics(np.radians(0.0), 0.0,10,5, 0.3, 0.03,)


 #torque = (self.l1.m * (self.l1.a**2) * theta_dt2)/3 + self.l2.m*(b**2 + self.l1.a**2)*theta_dt2 + (self.l2.a**2)*self.l2.m/12 + 2*self.l2.m*b*b_dt*theta_dt
            #torque = (((self.l1.m * (self.l1.a ** 2)) / 3) * theta_dt2 + self.l2.m * ((self.l1.a ** 2) + (b ** 2) + (self.l2.a ** 2) / 12) * theta_dt2) + (-self.l2.m*self.l2.a*b_dt2)
            #force = (self.l2.m*b_dt2) - self.l2.m*b_dt*(theta_dt**2)
            #force = (self.l2.m * (b_dt2 - self.l1.a*theta_dt2)) - self.l2.m * b_dt * (theta_dt ** 2)