from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel



# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginname')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account has not been activated by Admin.')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})


def user_view_dataset(request):
    from django.conf import settings
    import os
    import pandas as pd
    path = os.path.join(settings.MEDIA_ROOT,'uci_heart.csv')
    df = pd.read_csv(path)
    df = df.to_html
    return render(request, 'users/user_data_view.html',{'data':df})


def user_machine_learning(request):
    from .utility import MachineLearningUtility

    svc_accuracy, svc_precision, svc_recall, svc_f1score = MachineLearningUtility.calc_support_vector_classifier()
    j48_accuracy, j48_precision, j48_recall, j48_f1score = MachineLearningUtility.calc_j48_classifier()
    ann_accuracy, ann_precision, ann_recall, ann_f1score = MachineLearningUtility.calc_ann_model()
    my_accuracy, my_precision, my_recall, my_f1score = MachineLearningUtility.calc_proposed_model()

    j48_dict = {'j48_accuracy': j48_accuracy, 'j48_precision': j48_precision, "j48_recall": j48_recall,
               'j48_f1score': j48_f1score}
    ann_dict = {'ann_accuracy': ann_accuracy, 'ann_precision': ann_precision, 'ann_recall': ann_recall,
               'ann_f1score': ann_f1score}
    svc_dict = {'svc_accuracy': svc_accuracy, 'svc_precision': svc_precision, 'svc_recall': svc_recall,
                'svc_f1score': svc_f1score}
    my_dict = {'my_accuracy': my_accuracy, 'my_precision': my_precision, 'my_recall': my_recall,
               'my_f1score': my_f1score}

    return render(request, 'users/usermachinelearning.html',
                  {'j48': j48_dict, 'ann': ann_dict, "svc": svc_dict, 'my': my_dict})


def user_hidden_markov(request):
    from .utility import MachineLearningUtility
    hmm_result = MachineLearningUtility.calc_hmm_model()
    return render(request, 'users/user_hmm_result.html', {'hmm': hmm_result})


def user_predictions(request):
    if request.method=='POST':
        age = int(request.POST.get('age'))
        sex = int(request.POST.get('sex'))
        cp = int(request.POST.get('cp'))
        trestbps = int(request.POST.get('trestbps'))
        chol = int(request.POST.get('chol'))
        fbs = int(request.POST.get('fbs'))
        restecg = int(request.POST.get('restecg'))
        thalach = int(request.POST.get('thalach'))
        exang = int(request.POST.get('exang'))
        oldpeak = float(request.POST.get('oldpeak'))
        slope = int(request.POST.get('slope'))
        ca = int(request.POST.get('ca'))
        thal = int(request.POST.get('thal'))

        test_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak,slope,ca, thal]
        from .utility import MachineLearningUtility
        test_pred = MachineLearningUtility.test_user_date(test_data)
        if test_pred[0] == 0:
            rslt = False
        else:
            rslt = True
        return render(request, "users/predictions_form.html", {"test_data": test_data, "result": rslt})
    else:

        return render(request, 'users/predictions_form.html',{})
