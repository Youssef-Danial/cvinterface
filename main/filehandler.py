import profile
from django import forms
from main.models import postfile
import datetime
from django.utils import timezone
from django.core.files.storage import FileSystemStorage
import os
from django.core.files import File # for handling files
from django.core.exceptions import ValidationError
from django.core.validators import FileExtensionValidator

# format lists
audiolist = [".mp3"]
imagelist = ['.jpg','.png','.gif','.svg','.jpeg','.tiff']
videolist = [".mp4",".wmv",".avi",'.mkv']

# some validators
def file_size(value): # add this to some file where you can import it from
        limit = 20 * 1024 * 1024
        if value.size > limit:
            raise ValidationError('File too large. Size should not exceed 2 MiB.')
# FileExtensionValidator(allowed_extensions=['jpg','png','gif','svg','jpeg','tiff'])

# creating the form that will receive the files
class file_receiver(forms.Form):
    file = forms.ImageField(validators=[file_size, ])
class file_receiver_multi(forms.Form):
    file = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': True, "class":"btn btn-sm btn-info mr-4", "id":"myFile"}), required=False, validators=[file_size,FileExtensionValidator(allowed_extensions=['jpg','png','gif','svg','jpeg','tiff',"mp3","mp4","wmv","avi",'mkv'])])
# also should create file receiver for only audio files

def receive_file(request, state):

    if request.method == 'POST':
        # receving the file from the request
        form = file_receiver(request.POST, request.FILES)
        file_ = request.FILES['file']
        if form.is_valid():
            # converting the form to a dictionary
            file_received = form.cleaned_data
            # getting the current time for the uploaded_date for the file
            currentDateTime = datetime.datetime.now(tz=timezone.utc)
            # saving the file
            # getting user
            folder_path, access_path = make_dir(1, state = state)
            fs = FileSystemStorage(location=folder_path)
            file_path = os.path.join(folder_path, str(file_))
            # access_path
            access_path = os.path.join(access_path, str(file_))
            print(access_path)
            filesaver = file(file_name=str(file_path), uploaded_date=currentDateTime, file_url = access_path)
            fs.save(filesaver.file_name,file_)
            filesaver.save()
            return filesaver
    else:
        form = file_receiver()
        print("waiting for a file")
    return form

def receive_files(request, state, post=None):

    if request.method == 'POST':
        # receving the file from the request
        form = file_receiver_multi(request.POST, request.FILES)
        file_s = request.FILES.getlist('file')
        # creating a list of files
        filesuploadedlist = []
        if form.is_valid():
            for file_ in file_s:
                # converting the form to a dictionary
                file_received = form.cleaned_data
                # getting the current time for the uploaded_date for the file
                currentDateTime = datetime.datetime.now(tz=timezone.utc)
                # saving the file
                # getting user
                #u = get_user(request)
                folder_path, access_path = make_dir(1, state = state)
                fs = FileSystemStorage(location=folder_path)
                file_path = os.path.join(folder_path, str(file_))
                # getting file type
                file_type = get_file_type(get_file_extension(str(file_path)))
                # access_path
                access_path = os.path.join(access_path, str(file_))
                print(access_path)
                if state == "post":
                    filesaver = postfile(file_name=str(file_path), uploaded_date=currentDateTime, file_url = access_path, file_extension = get_file_extension(str(file_path)), file_type = file_type)
                fs.save(filesaver.file_name,file_)
                filesaver.save()
                filesuploadedlist.append(filesaver)
            return filesuploadedlist
    else:
        form = file_receiver_multi()
        print("waiting for a files")
    return form

def make_dir(user_id, state=None):
    user_id = str(user_id)
    # this line need to be changed every time the website files be moved
    static_path = "static/files"
    save_path = "/files"
    base = os.getcwd()
    all_path = os.path.join(base, static_path)
    new_path = os.path.join(all_path, user_id)
    # creating a directory if it does not exist
    try:
        if not os.path.exists(new_path):
            os.mkdir(new_path)
    except:
        pass
    # to store profile files like : (profile picture and background picture)
    if state == "post":
        profile_path = os.path.join(new_path, "post")
        try:
            if not os.path.exists(profile_path):
                 os.mkdir(profile_path)
        except:
            pass
        temp_path =  os.path.join(save_path, user_id)
        access_path = os.path.join(temp_path, "post")
        return  profile_path, access_path
    else:
        # to be continue
        return None, None
# this function take a file name and retrieve the file extension
def get_file_extension(filename):
    extension = os.path.splitext(filename)[1]
    return extension

def get_file_type(extension):
    file_type = ""
    if extension in imagelist:
        file_type = "image"
    else:
        file_type = "unkow" # unknown
    return file_type

def load_file(file_to_load):
    pass
