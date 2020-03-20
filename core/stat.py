import json




def get_ids():
    params = {"intention_set": ["departmentAmong", "departmentSubset", "departmentConfirm", "department", "content",
                                "keyword_medicine", "keyword_treatment", "keyword_symptom", "doctor", "hospital",
                                "hospitalQuality", "hospitalDepartment", "recentHaoyuanTime", "doctorQuality",
                                "haoyuanRefresh", "register", "keyword_disease", "keyword_hospital",
                                "keyword_department", "keyword_doctor", "customerService", "greeting", "guide",
                                "corpusGreeting", "sensitive", "auto_diagnose", "other", "keyword_city",
                                "keyword_province", "keyword_body_part", "keyword_examination", "keyword_medical_word"],
              "mode": "xwyz",
              "q": "头痛挂什么科",
              "accompany_intention_set": ["patient_group"]}
    with open(file='C:\\Users\\weiyi\\Desktop\\234.txt', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            data = line.strip().split(',')[0][1:-1]
            print(data)


def p():
    with open(file='C:\\Users\\weiyi\\Desktop\\234.txt', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            data = line.strip().split(',')[0][1:-1]
            print(data)


def regex_f():
    import re
    data = '预约挂号有问题，报错原因是：上海交通大学医学院附属新华医院-儿神经外科-陈若平，倪演哲-310***********2919-189*****305。   就诊人身份证年龄超过排班要求的年龄上限。 (班次信息：20191224 上午 09:30-09:59)	'
    pattern = re.compile(r'.*挂号有问题[，,]报错原因是.*')
    result = pattern.match(data)
    print(result)

if __name__ == '__main__':
    # get_ids()
    regex_f()