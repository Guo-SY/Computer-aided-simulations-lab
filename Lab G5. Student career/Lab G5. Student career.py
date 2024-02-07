import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from scipy.stats import bernoulli

import seaborn as sns


                              #----------------------------Innitiation-----------------------------#
#innitiation
GT = 0                                   #Graduation Time
total_grade_score = 0                    #The Final Graudation grades

avg = 0                                
max_attend_times = 4
graduation = False

num_pass = 0
attend_counter = 0

pass_course_list = []


class Course():
    def __init__(self,course_Name):
        self.course_pass = False
        self.course_grade = 0
        self.counter_attendOfExam = 1
        self.course_Name = course_Name

class Student():
    def __init__(self, student_ID):
        self.student_ID = student_ID
        self.graduation_year = 0
        self.final_grade = 0 
        self.course_name_list = list("ABCDEFGHIJ")
        self.mark = None
        self.pass_course_list = []


                             #----------------------------Source Dataset-----------------------------#
A = "Data Management and visualization"
B = "Data Science lab:process and methods"
C = "Data Ethics and Data Protection"
D = "Distributed architecture for big data processing and analythics"
E = "Machine learning and Deep learning"
F = "Mathematics in Machine Learning"
G = "Statistical Methods in data science"
H = "Decision making and optimization"
I = "Data Science lab:programming" 
J = "Innovation management"
K = "Machine learning for IOT"
L = "Applied data science project"

courses_data = {
            "A":{18: 26, 19: 10, 20: 7, 21: 7, 22: 3, 23: 10, 24: 6, 25: 20, 26: 14, 27: 22, 28: 17, 29: 18, 30: 6},
            "B":{18: 7, 19: 10, 20: 10, 21: 12, 22: 13, 23: 13, 24: 17, 25: 12, 26: 24, 27: 14, 28: 10, 29: 14, 30: 23},
            "C":{18: 16, 19: 10, 20: 11, 21: 16, 22: 12, 23: 13, 24: 15, 25: 12, 26: 9, 27: 8, 28: 10, 29: 1, 30: 0},
            "D":{18: 17, 19: 1, 20: 4, 21: 5, 22: 2, 23: 9, 24: 3, 25: 5, 26: 7, 27: 14, 28: 12, 29: 16, 30: 44},
            "E":{18: 13, 19: 0, 20: 5, 21: 7, 22: 5, 23: 8, 24: 9, 25: 12, 26: 10, 27: 22, 28: 27, 29: 28, 30: 57}, 
            "F":{18: 8, 19: 15, 20: 9, 21: 11, 22: 9, 23: 9, 24: 8, 25: 6, 26: 7, 27: 6, 28: 4, 29: 9, 30: 14},
            "G":{18: 21, 19: 16, 20: 10, 21: 7, 22: 5, 23: 10, 24: 4, 25: 4, 26: 6, 27: 11, 28: 13, 29: 7, 30: 19},
            "H":{18: 5, 19: 3, 20: 2, 21: 0, 22: 3, 23: 0, 24: 6, 25: 5, 26: 1, 27: 4, 28: 2, 29: 3, 30: 14},
           "I":{18: 7, 19: 10, 20: 10, 21: 12, 22: 13, 23: 13, 24: 17, 25: 12, 26: 24, 27: 14, 28: 10, 29: 14, 30: 23},
            "J":{18: 10, 19: 4, 20: 12, 21: 9, 22: 7, 23: 11, 24: 4, 25: 8, 26: 7, 27: 9, 28: 7, 29: 3, 30: 8},
            "K":{18: 2, 19: 4, 20: 1, 21: 2, 22: 3, 23: 3, 24: 2, 25: 6, 26: 2, 27: 4, 28: 2, 29: 4, 30: 24},
            "L":{18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 1, 29: 9, 30: 16}
                }


# Give a Pass Probabilities of Exams
P_J =  0.65                               #np.random.uniform(0.60, 1)  # Replace with the actual probability P(J)
P_B_given_J = 0.55
P_C_given_J = 0.55
P_D_given_J = 0.55
P_A_given_J = 0.55
P_E_given_J = 0.55
P_G_given_J = 0.55
P_F_given_J = 0.55
P_H_given_J = 0.55
P_I_given_J = 0.55
P_K_given_J = 0.55
P_L_given_J = 0.55

# Calculate P(B, C, D, A, E, F, G, H, I|J)
P_B_C_D_A_E_F_G_H_I_K_L_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J * P_I_given_J* P_K_given_J* P_L_given_J
P_B_C_D_A_E_F_G_H_I_K_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J * P_I_given_J* P_K_given_J
P_B_C_D_A_E_F_G_H_I_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J * P_I_given_J
P_B_C_D_A_E_F_G_H_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J
P_B_C_D_A_E_F_G_H_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J
P_B_C_D_A_E_F_G_H_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J * P_H_given_J
P_B_C_D_A_E_F_G_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J * P_G_given_J
P_B_C_D_A_E_F_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J * P_F_given_J
P_B_C_D_A_E_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J * P_E_given_J
P_B_C_D_A_given_J = P_B_given_J * P_C_given_J * P_D_given_J * P_A_given_J
P_B_C_D_given_J = P_B_given_J * P_C_given_J * P_D_given_J
P_B_C_given_J = P_B_given_J * P_C_given_J


# Calculate P(B, C, D, A, E, F, G, H, I)
# The probability of the student passing all exams.
P_B_C_D_A_E_F_G_H_I_k_L = 0.3
P_B_C_D_A_E_F_G_H_I_K = 0.35
P_B_C_D_A_E_F_G_H_I = 0.4
P_B_C_D_A_E_F_G_H = 0.47
P_B_C_D_A_E_F_G = 0.50
P_B_C_D_A_E_F = 0.53
P_B_C_D_A_E = 0.55
P_B_C_D_A = 0.57
P_B_C_D = 0.60
P_B_C = 0.62
P_B = 0.65

# Calculate P(J|B, C, D, A, E, F, G, H, I)
P_J_given_B_C_D_A_E_F_G_H_I_K_L = (P_B_C_D_A_E_F_G_H_I_K_L_given_J * P_J) / P_B_C_D_A_E_F_G_H_I_k_L
P_J_given_B_C_D_A_E_F_G_H_I_K = (P_B_C_D_A_E_F_G_H_I_K_given_J * P_J) / P_B_C_D_A_E_F_G_H_I_K
P_J_given_B_C_D_A_E_F_G_H_I = (P_B_C_D_A_E_F_G_H_I_given_J * P_J) / P_B_C_D_A_E_F_G_H_I
P_J_given_B_C_D_A_E_F_G_H = (P_B_C_D_A_E_F_G_H_given_J * P_J) / P_B_C_D_A_E_F_G_H
P_J_given_B_C_D_A_E_F_G = (P_B_C_D_A_E_F_G_given_J * P_J) / P_B_C_D_A_E_F_G
P_J_given_B_C_D_A_E_F = (P_B_C_D_A_E_F_G_given_J * P_J) / P_B_C_D_A_E_F
P_J_given_B_C_D_A_E = (P_B_C_D_A_E_given_J * P_J) / P_B_C_D_A_E
P_J_given_B_C_D_A = (P_B_C_D_A_given_J * P_J) / P_B_C_D_A
P_J_given_B_C_D = (P_B_C_D_given_J * P_J) / P_B_C_D
P_J_given_B_C = (P_B_C_given_J * P_J) / P_B_C
P_J_given_B = (P_B_given_J * P_J) / P_B



Pass_exam_probabilities = {
                                1: P_J,
                                2: P_J_given_B,
                                3: P_J_given_B_C,
                                4: P_J_given_B_C_D,
                                5: P_J_given_B_C_D_A,
                                6: P_J_given_B_C_D_A_E,
                                7: P_J_given_B_C_D_A_E_F,
                                8: P_J_given_B_C_D_A_E_F_G,
                                9: P_J_given_B_C_D_A_E_F_G_H,
                                10:P_J_given_B_C_D_A_E_F_G_H_I,
                                11:P_J_given_B_C_D_A_E_F_G_H_I_K,
                                12:P_J_given_B_C_D_A_E_F_G_H_I_K_L

                                }


                          #--------------------step 1: Create a random course grade---------------------------#

# # Print the value associated with key 10

def pass_exam_probability(attendNumExams):
    probability = Pass_exam_probabilities[attendNumExams]
    return probability if 0 <= probability <= 1 else None

'''The pass probability of exam will decresee acooeding to the number of attend exames'''

def course_grade_generate(course_Name):
    course_Name = str(course_Name)
    grade_distribution = courses_data[course_Name]
    course_grade = [i for i in grade_distribution.keys()]
    course_counts = [j for j in grade_distribution.values()]

    # Check if either course_grade or course_counts is empty
    if not course_grade or not course_counts:
        print(f"Invalid grade distribution for {course_Name}")
        return None

    # Generate random samples based on the given distribution
    num_samples = 100  
    course_grade_list = np.random.choice(course_grade, size=num_samples, p=course_counts/np.sum(course_counts))
    
    return course_grade_list
'''There are different grade graduatitions with different courses of MCs'''



                     #--------------------step 2: Creat a course_list ---------------------------#
def create_course_list(courses_name_list):  
    # courses_list = list("ABCDEFGHIJ")
    np.random.shuffle(courses_name_list)
    

    course_list = []
    for name in courses_name_list:
        course = Course(name)
        course_list.append(course)

    return course_list


def create_course_list(courses_name_list):  
    # courses_list = list("ABCDEFGHIJ")
    np.random.shuffle(courses_name_list)
    

    course_list = []
    for name in courses_name_list:
        course = Course(name)
        course_list.append(course)

    return course_list

def single_student_graduation(courses_name_list,courses_data, Pass_exam_probabilities):
    graduation = False
    avg = 0
    final_grade = 0
    max_attend_times = 4   #Students may take up to 4 exams per year

    num_pass = 0
    total_grade_score = 0
    attend_counter = 0
    pass_course_list = []
    GT = 0

    course_list = create_course_list(courses_name_list)
    

    while len(course_list) > 0 and not graduation:

        loop_counter = 0
        # attend_course_list = np.random.choice(course_list, size=max_attend, replace=False)

        while loop_counter <= max_attend_times and not graduation:
            
            # The student could decide how many exams they will attend 
            attendExamNumber  = 0
            attend_course_list = course_list



            #counter all courses from the attend_course_list which the student will attend
            for course in attend_course_list:  
                # The pass probability of exam will decrease with the increasing number of exam
                attendExamNumber += 1
                prob = pass_exam_probability(attendExamNumber) 

                # # Generate Bernoulli-distributed values
                if prob is not None:
                    # Generate Bernoulli-distributed values
                    course.course_pass = np.random.choice([0, 1], size=1, p=[1 - prob, prob])
                    
                else:
                    print("Invalid probability")

                # after get the course.course_pass, it will update information 
                
                
                if course.course_pass:
                    # if student pass the exm, he will get a score.
                    # choose a specific grade for a course from the list of course_grade_list
                    grades_list_of_course = course_grade_generate(course.course_Name)
                    specific_course_grade = np.random.choice(grades_list_of_course)
                    course.course_grade = specific_course_grade


                    # when they get a score, they will decide if accept it
                    random_p = np.random.random()
                    rv = bernoulli(random_p)
                    # Generate random samples
                    accept =  rv.rvs(size=1)
                    
                    if accept == 1:
                        num_pass += 1
                        total_grade_score += course.course_grade
                        pass_course_list.append(course)

                        # avg = total_grade_score/num_pass
                        attend_course_list.remove(course)

                    else: 
                        # if they don not want to accept the score
                        graduation = False
                        course.counter_attendOfExam += 1
                        course.course_grade = 0   #random_course_grade
                        course.course_pass = 0

                else:
                    graduation = False
                    course.counter_attendOfExam += 1
                    # choose a random from the list of all_course_grade
                    random_course_grade = 0
                    course.course_grade = random_course_grade

                    course.course_pass = 0

            # print(f' \n For the {loop_counter}th loop, the number of pass course: {len(pass_course_list)}')
            loop_counter += 1


        if len(course_list) > 0:
            graduation = False
            GT += 1

        else:
            graduation = True
            GT += 1


        # compute the final grade gor student
        if num_pass > 0:
            avg = total_grade_score/num_pass
        else:
            avg = 0

        grade = (avg/30)*110 + np.random.uniform(0,4) + np.random.uniform(0,2) + np.random.uniform(0,2)
        final_grade = round(grade,3)
        
    return GT,avg,final_grade,pass_course_list

                          #--------------------step 3: Create a series of student---------------------------#

def multi_student_graduation(student_number,courses_name_list):

    total_graduatin_grades = 0
    total_graduatin_year = 0
    avg_graduatin_year = 0
    avg_graduatin_grade = 0

    student_list = []
    

    #  ist of course_grade for all students
    # all_course_grade = course_grade_generate(total_course)


   #Creat a series of student 
    for i in np.linspace(0, student_number - 1, student_number, dtype=int):
        student = Student(i)
        student_list.append(student)

    for student in student_list:
        total_course_name = student.course_name_list
        GT,avg,final_grade,pass_course_list = single_student_graduation(total_course_name,courses_data, Pass_exam_probabilities)

        
        # GT,avg,final_grade,pass_course_list = single_student_graduation(total_course,all_course_grade)

        if final_grade > 112.5:
            student.mark = '110L'

        else:
            student.mark = str(final_grade)

        student.final_grade = final_grade
        student.graduation_year = GT
        student.pass_course_list = pass_course_list


        total_graduatin_grades += final_grade
        total_graduatin_year += GT

    avg_graduatin_year = total_graduatin_year/len(student_list)
    avg_graduatin_grade = total_graduatin_grades/len(student_list)

    return avg_graduatin_year,avg_graduatin_grade,student_list


                              #----------------------------Creat a Event Loop -----------------------------#

# for a single student
all_course_grade = courses_data
all_course_pass_probability = Pass_exam_probabilities
courses_name_list = list("ABCDEFGHIJKL")

GT,avg,final_grade,pass_course_list = single_student_graduation(courses_name_list, courses_data, all_course_pass_probability)

print('\n For a single student')
print('\n Congratulations on Graduation!!!!')
print(f'\n The final grade is :{final_grade}')
# print(f'\n The final num_pass is: {num_pass}')
print(f'\n The graduation year is: {GT}')
# print(f'\n The loop counter is: {loop_counter}')


# for many students
student_number = 500


avg_graduatin_year,avg_graduatin_grade,student_list = multi_student_graduation(student_number,courses_name_list)

print(f'The average graduation year: {avg_graduatin_year}')
print(f'The average graduation grade: {avg_graduatin_grade}')
print(f'The number of attended exam per student : {len(student_list[0].pass_course_list)}')


                            #--------------Show the grades distribution--------------------#

# Given data
courses_name_list = list("ABCDEFGHIJ")

# x = np.arange(0, 101, 1)

# Plot the grade distribution and corresponding probabilities for each course
plt.figure(figsize=(12,8))

for index, course_name in enumerate(courses_name_list):
    course_name = str(course_name)
    grade_distribution = courses_data[course_name]
    course_grade = np.array([i for i in grade_distribution.keys()])
    course_counts = np.array([j for j in grade_distribution.values()])
    grade_probability = course_counts / np.sum(course_counts)

    # Plot the grade distribution
    plt.plot(course_grade, course_counts, label=f'{course_name} - Distribution', marker='o')

    # Plot the corresponding probabilities using a scatter plot
    plt.scatter(course_grade, grade_probability, marker='o')

    # Add text annotations for each point in the scatter plot
    for grade, probability, counts in zip(course_grade, grade_probability, course_counts):
        plt.annotate(text=f'{str(round(probability, 2))}', xy=(grade, counts),
                     xytext=(5, 5), textcoords='offset points', ha='center', va='bottom', fontsize=8)


plt.xticks(course_grade)
plt.title('Grade Distribution and Probabilities for Each Course')
plt.xlabel('Course grade')
plt.ylabel('Count / Probability')
plt.grid()
plt.legend()
plt.show()


                #--------------------------Compute the average number of attended exams---------------------------------#
                 
distribution_student_attendedExams = np.array( [(student_list[i].student_ID, student_list[i].pass_course_list) 
                                                  for i in range(len(student_list))])
# print(len(distribution_student_attendedExams))

student_ID_list = distribution_student_attendedExams[:, 0]   #len = 20
all_passed_course = distribution_student_attendedExams[:, 1]  #len = 20

# avg_attendedExams_list = [sum(passed_course[i].counter_attendOfExam)/len(passed_course ) for i in len(all_passed_course)]

# print(len(student_ID_list))
# print(len(all_passed_course))
# zip(student_ID_list,avg_attendedExams_list)

                #--------------------------------------------------------------------------------#

avg_single_attendedExams = 0
attendedExams = 0

single_counter_list = []
students_avg_attendedExams = []

for i in range(len(student_ID_list)):
    single_attendedExams = 0

    for j in range(len(all_passed_course[i])):
        single_attendedExams += all_passed_course[i][j].counter_attendOfExam

    avg_single_attendedExams = round(single_attendedExams/len(all_passed_course[i]),4)

    students_avg_attendedExams.append((student_ID_list[i],avg_single_attendedExams))

# print(students_avg_attendedExams)

students_avg_attendedExams = np.array(students_avg_attendedExams)
attendedExams = sum(students_avg_attendedExams[: , 1])

avg_single_attendedExams = attendedExams/ len(students_avg_attendedExams)

# print(avg_single_attendedExams)

                #--------------------------------------------------------------------------------#
student_graduatioin = np.array( [(student_list[i].student_ID,student_list[i].graduation_year,student_list[i].final_grade) for i in range(len(student_list))])
student_ID_list = student_graduatioin[:, 0]   #len = 20
student_graduatioin_year = student_graduatioin[:, 1]  #len = 20
student_graduatioin_grade = student_graduatioin[:, 2]  #len = 20





# Create a figure with two subplots
fig, ax = plt.subplots(3,1, figsize=(20,12))
# plt.hist(students_avg_attendedExams[: , 0],students_avg_attendedExams[: , 1], bins=len(students_avg_attendedExams), density=True, alpha=0.7, color='blue', label='Monte Carlo Samples')
ax[0].plot(student_ID_list,student_graduatioin_year, color='red', label='graduatioin_year')
ax[0].set_title('The Simulation for graduation year of students')
# plt.text(random_course_grade, counts/np.sum(counts), str(counts/np.sum(counts)), ha='center', va='bottom')
student_ID_labels = [f'student_{str(id+1)}' for id in student_ID_list]
# ax[0].set_xticks(student_ID_list)
# ax[0].set_xticklabels(student_ID_labels, rotation=45, ha='right')

ax[0].set_xlabel('Student_ID')
ax[0].set_ylabel('graduatioin_year')
ax[0].grid()



ax[1].plot(student_ID_list,student_graduatioin_grade, color='blue', label='graduatioin_grade')
ax[1].set_title('The Simulation for final graduation grades of students')
# plt.text(random_course_grade, counts/np.sum(counts), str(counts/np.sum(counts)), ha='center', va='bottom')
student_ID_labels = [f'student_{str(id+1)}' for id in student_ID_list]
# ax[1].set_xticks(student_ID_list)
# ax[1].set_xticklabels(student_ID_labels, rotation=45, ha='right')

ax[1].set_xlabel('Student_ID')
ax[1].set_ylabel('graduatioin_grade')
ax[1].grid()
ax[1].legend()

# plt.figure(figsize=(12, 8))  
ax[2].plot(students_avg_attendedExams[: , 0],students_avg_attendedExams[: , 1], color='red', label='Given Data')
ax[2].set_title('The Simulation for average attended exams of students')


ax[2].set_xlabel('Student_ID')
ax[2].set_ylabel('Avg Number of Attended Exams')
ax[2].grid()
# ax[2].legend()
# ax[2].show()



# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()




                                 #------------find some interseting relation-----------------#
''' you could cancle this part code'''

# Combine lists into a DataFrame
df = pd.DataFrame({
                    # 'Student_ID': student_ID_list, 
                    'average attended Exams': students_avg_attendedExams[:,1], 
                    'graduatioin_year':  student_graduatioin_year, 
                    'graduatioin_grade': student_graduatioin_grade
                    
                    })


# Compute the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()






               #-------------------------------The final results----------------------------------------#
mean_avg_attendedExams = df['average attended Exams'].mean()
mean_graduatioin_year = df['graduatioin_year'].mean()
mean_graduatioin_grade = df['graduatioin_grade'].mean()

variance_avg_attendedExams = df['average attended Exams'].var()
variance_graduatioin_year = df['graduatioin_year'].var()
variance_graduatioin_grade = df['graduatioin_grade'].var()

def accuracy(data):
    confidence_level = 0.95

    sample_size = len(data)
    sample_mean = data.mean() 
    squared_deviations =data.var()

    confidence_interval = stats.norm.interval(confidence_level, loc=sample_mean, scale=squared_deviations/np.sqrt(sample_size))
    lower_bound, upper_bound = confidence_interval
    relative_error = ((upper_bound - lower_bound )/2)/sample_mean
    
    return (1-relative_error)

accuracy_avg_attendedExams     = accuracy(df['average attended Exams'])
accuracy_graduatioin_year      = accuracy(df['graduatioin_year'])
accuracy_graduatioin_grade     = accuracy(df['graduatioin_grade'])          



# Create a DataFrame with statistics
df_statistic = pd.DataFrame({
    'Statistic': ['mean', 'variance', 'accuracy'],
    'average attended Exams': [mean_avg_attendedExams, variance_avg_attendedExams, accuracy_avg_attendedExams],
    'graduation_year': [mean_graduatioin_year, variance_graduatioin_year, accuracy_graduatioin_year ],
    'graduation_grade': [mean_graduatioin_grade, variance_graduatioin_grade, accuracy_graduatioin_grade]
})

# Set 'Student_ID' as the index
df_statistic.set_index('Statistic', inplace=True)

# Display the resulting DataFrame with means and variances
print(df_statistic)