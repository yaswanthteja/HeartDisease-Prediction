from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel


# Create your views here.

def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')
        elif usrid == 'Admin' and pswd == 'Admin':
            return render(request, 'admins/AdminHome.html')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def ViewRegisteredUsers(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/RegisteredUsers.html', {'data': data})


def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

def Admin_view_metrics(request):
    from users.utility import MachineLearningUtility

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

    return render(request, 'admins/metrics_results.html',
                  {'j48': j48_dict, 'ann': ann_dict, "svc": svc_dict, 'my': my_dict})

