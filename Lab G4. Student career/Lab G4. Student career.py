''' I must apologize for this part of my code. Because I wanted to generate heatmaps to show the correlation 
between different parameters.
Before executing this code, you need to install the seaborn package. 
If you don't want to install seaborn, please cancel the last part of the code.'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math

from scipy.stats import bernoulli




                               #----------------------------Innitiation-----------------------------#
#innitiation
GT = 0
total_grade_score = 0
total_course = 12
avg = 0
max_attend = 4
graduation = False

num_pass = 0
attend_counter = 0


pass_course_list = []




class Course():
    def __init__(self,course_ID):
        self.course_pass = True
        self.course_grade = 0
        self.counter_attendOfExam = 1
        self.course_ID = course_ID

class Student():
    def __init__(self, student_ID):
        self.student_ID = student_ID
        self.graduation_year = 0
        self.final_grade = 0 
        self.total_course = 12
        self.mark = None
        self.pass_course_list = []




                          #--------------------step 1: Create a random course grade---------------------------#
def course_grade_generate(total_course):

    num_samples = total_course

    # Given data
    data = np.array([(18, 87), (19, 62), (20, 74), (21, 55), (22, 99), (23, 94),
                    (24, 117), (25, 117), (26, 136), (27, 160), (28, 215), (29, 160), (30, 473)])

    # Extract age and count values
    course_grade, counts = data[:, 0], data[:, 1]
    
    
    greater18_course_grade = [data for data in np.random.choice(course_grade, size=num_samples, p=counts/np.sum(counts))]
    less18_random_data = [data for data in np.random.randint(0, 18, num_samples)]


    # Combine lists using the + operator
    all_course_grade =  greater18_course_grade + less18_random_data


    # Shuffle the combined list
    np.random.shuffle(all_course_grade)


    return all_course_grade



                     #--------------------step 2: Creat a course_list ---------------------------#

def create_course_list(total_course,all_course_grade):  
    course_list = []
    for i in np.linspace(0, total_course - 1, total_course, dtype=int):
        course = Course(i)

        # choose a random from the list of all_course_grade
        random_course_grade = np.random.choice(all_course_grade)
        course.course_grade = random_course_grade


        # course.course_ID = i

        if course.course_grade < 18:
            course.course_pass = False 
        else:
            course.course_pass = True


        course_list.append(course)

        # #    check point  #
        # print(len(course_list))
        # print(course_list[0].course_ID)
        # print(course_list[0].course_grade)
        # print(course_list[0].course_pass)
        # print(course_list[0].counter_attendOfExam)

        # print(course_list[1].course_ID)
        # print(course_list[1].course_grade)
        # print(course_list[1].course_pass)
        # print(course_list[1].counter_attendOfExam)

    return course_list


def single_student_graduation(total_course,all_course_grade):
    graduation = False
    avg = 0
    final_grade = 0
    
    max_attend = 4
    graduation = False

    num_pass = 0
    total_grade_score = 0
    attend_counter = 0
    pass_course_list = []
    GT = 0
    # total_course = 12

    course_list = create_course_list(total_course,all_course_grade)

    while len(course_list) > 0 and not graduation:

        loop_counter = 0

        while loop_counter <= max_attend and not graduation:
            for course in course_list:
                if course.course_pass:
                    # Define the probability of success (p)
                    random_p = np.random.random()

                    # Create a Bernoulli random variable
                    rv = bernoulli(random_p)

                    # Generate random samples
                    accept =  rv.rvs(size=1)
                    
                    if accept == 1:
                        num_pass += 1
                        total_grade_score += course.course_grade
                        pass_course_list.append(course)

                        # avg = total_grade_score/num_pass
                        course_list.remove(course)

                    else:
                        graduation = False
                        course.counter_attendOfExam += 1

                        # choose a random from the list of all_course_grade
                        random_course_grade = np.random.choice(all_course_grade)
                        course.course_grade = random_course_grade

                        course.course_pass = course.course_grade >= 18

                else:
                    graduation = False
                    course.counter_attendOfExam += 1
                    # choose a random from the list of all_course_grade
                    random_course_grade = np.random.choice(all_course_grade)
                    course.course_grade = random_course_grade

                    course.course_pass = course.course_grade >= 18

            # print(f' \n For the {loop_counter}th loop, the number of pass course: {len(pass_course_list)}')
            loop_counter += 1


        if len(course_list) > 0:
            graduation = False
            GT += 1

        else:
            graduation = True
            GT += 1


        # compute the final grade gor student
        avg = total_grade_score/num_pass
        grade = (avg/30)*110 + np.random.uniform(0,4) + np.random.uniform(0,2) + np.random.uniform(0,2)
        final_grade = round(grade,3)



        # print('\n Congratulations on Graduation!!!!')
        # print(f'\n The final grade is :{avg}')
        # print(f'\n The final num_pass is: {num_pass}')
        # print(f'\n The graduation year is: {GT}')
        # print(f'\n The loop counter is: {loop_counter}')
        
    return GT,avg,final_grade,pass_course_list
            




                          #--------------------step 3: Create a series of student---------------------------#


 
def multi_student_graduation(student_number,total_course):

    total_graduatin_grades = 0
    total_graduatin_year = 0
    avg_graduatin_year = 0
    avg_graduatin_grade = 0

    student_list = []
    

    # Generate a list of course_grade for all students
    all_course_grade = course_grade_generate(total_course)

    for i in np.linspace(0, student_number - 1, student_number, dtype=int):
        student = Student(i)
        student_list.append(student)

    for student in student_list:
        total_course = student.total_course
        GT,avg,final_grade,pass_course_list = single_student_graduation(total_course,all_course_grade)

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
# for a single student

total_course = 12
final_grade = 0
all_course_grade = course_grade_generate(total_course)

GT,avg,final_grade,pass_course_list = single_student_graduation(total_course,all_course_grade)

print('\n For a single student')
print('\n Congratulations on Graduation!!!!')
print(f'\n The final grade is :{final_grade}')
# print(f'\n The final num_pass is: {num_pass}')
print(f'\n The graduation year is: {GT}')
# print(f'\n The loop counter is: {loop_counter}')


# for many students
student_number = 2000
total_course = 12


avg_graduatin_year,avg_graduatin_grade,student_list = multi_student_graduation(student_number,total_course)

print(f'The average graduation year: {avg_graduatin_year}')
print(f'The average graduation grade: {avg_graduatin_grade}')
print(f'The number of attended exam per student : {len(student_list[0].pass_course_list)}')







                             #--------------Show the grades distribution--------------------#
# Given data
data = np.array([(18, 87), (19, 62), (20, 74), (21, 55), (22, 99), (23, 94),
                 (24, 117),(25,117), (26, 136), (27, 160), (28, 215), (29, 160), (30, 473)])

# Extract age and count values
course_grade, counts = data[:, 0], data[:, 1]

# Generate random samples based on the given distribution
num_samples = 10000  # You can adjust the number of samples
random_course_grade = np.random.choice(course_grade, size=num_samples, p=counts/np.sum(counts))

# Plot the histogram of the generated samples
plt.figure(figsize=(12, 8))  # Corrected line
plt.hist(random_course_grade, bins=len(course_grade), density=True, alpha=0.7, color='blue', label='Monte Carlo Samples')
plt.plot(course_grade, counts/np.sum(counts), color='red', label='Given Data')

# Add specific data on top of each bar
for value, count in zip(course_grade, counts):
    probability = round(count/np.sum(counts),3)
    plt.text(value, probability, f"{count/np.sum(counts):.3f}", ha='center', va='bottom', fontsize=12)

plt.title('Monte Carlo Simulation of Grades Distribution')
# plt.text(random_course_grade, counts/np.sum(counts), str(counts/np.sum(counts)), ha='center', va='bottom')
plt.xlabel('Course grade')
plt.ylabel('Probability')
plt.legend()
plt.show()


                 #--------------------------Compute the average number of attended exams---------------------------------#
distribution_student_attendedExams = np.array( [(student_list[i].student_ID,student_list[i].pass_course_list) for i in range(len(student_list))])
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
plt.figure(figsize=(12, 8))  # Corrected line

plt.plot(students_avg_attendedExams[: , 0],students_avg_attendedExams[: , 1], color='red', label='Given Data')

# Add specific data on top of each bar
# for student_id, student_grades in zip(students_avg_attendedExams[: , 0], students_avg_attendedExams[: , 1]):

#     plt.text(student_id, student_grades, f"{student_grades:.3f}", ha='center', va='bottom', fontsize=12)

plt.title('The Simulation for average attended exams of students')
# plt.text(random_course_grade, counts/np.sum(counts), str(counts/np.sum(counts)), ha='center', va='bottom')
# student_ID_labels = [f'student_{str(id+1)}' for id in students_avg_attendedExams[: , 0]]
# plt.xticks(students_avg_attendedExams[: , 0], student_ID_labels, rotation=45, ha='right')

plt.xlabel('Student_ID')
plt.ylabel('Avg Number of Attended Exams')
plt.grid()
plt.legend()
plt.show()


                #--------------------------------------------------------------------------------#
student_graduatioin = np.array( [(student_list[i].student_ID,student_list[i].graduation_year,student_list[i].final_grade) for i in range(len(student_list))])
student_ID_list = student_graduatioin[:, 0]   #len = 20
student_graduatioin_year = student_graduatioin[:, 1]  #len = 20
student_graduatioin_grade = student_graduatioin[:, 2]  #len = 20


# Create a figure with two subplots
fig, ax = plt.subplots(2,1, figsize=(18,12))
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

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the plot
plt.show()



#---------------------------------------FBI warning--------------------------------#
#---------------------------------------FBI warning--------------------------------#
# !pip install seaborn

                                 #------------find some interseting relation-----------------#
''' you could cancle this part code'''

import pandas as pd
import seaborn as sns
# Combine lists into a DataFrame
df = pd.DataFrame({
    'Student_ID': student_ID_list, 
    'average attended Exams': students_avg_attendedExams[:,1], 
    'graduatioin_year':  student_graduatioin_year, 
    'graduatioin_grade': student_graduatioin_grade})
# Display the DataFrame
# print(df)


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