from numpy import *



def compute_error(b,m ,points):
    total_error = 0
    for i in range(0 , len(points)):
        x = points[i , 0 ]
        y = points[ i ,1]
        total_error += (y - ( m * x +b))**2
    return  total_error/float(len(points))


def step_gradient(b_current , m_current , points ,learning_rate):
    #gradient deescent
    b_gradient = 0
    m_gradient =0
    N =float(len(points))
    for i  in range(0,len(points)):
        x  = points[i , 0]
        y = points[i ,1]
        b_gradient += -(2/N)*(y - ((m_current * x)+b_current))
        m_gradient += -(2/N)*x*(y - ((m_current * x)+b_current))
    new_b = b_current -(learning_rate * b_gradient)
    new_m = m_current -(learning_rate * m_gradient)
    return [new_b , new_m]



def gradient_descent_runner(points , learning_rate , starting_b , starting_m ,num_iterations):
    b = starting_b
    m =starting_m
    for i in range(num_iterations):
        b, m =step_gradient(b , m ,array(points) , learning_rate)
        return [b,m]



def run():
    points = genfromtxt('data.csv' , delimiter=',')
    """hyper parameters"""
    learning_rate =0.001
    """y=mx+c"""
    initial_b =0
    initial_m =0
    num_iterations =1000
    print("Staring Gradient descent at b ={0} , m={1} ,error = {2}".format(initial_b , initial_m  ,compute_error(initial_b , initial_m ,points)))
    print("<<<<<<<<<<<<<<<RUNNINGGGGG>>>>>>>>>>>>>")

    [b,m] = gradient_descent_runner(points , learning_rate , initial_b , initial_m , num_iterations)
    print("After {0} iterations b ={1} , m={2} , error={3}".format(num_iterations , b ,m ,compute_error(b ,m ,points)) )
    print(b)
    print(m)



if __name__ == '__main__':
    run()