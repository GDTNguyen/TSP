# TSP

This project attempts to solve Travelling Salesman Problems using the Backbone method. The application is a Web Application that gives the user the opportunity to create their own problems for the algorithm to solve, and it also allows the user to solve it too.

## Installation and Usage

The OSXExecu and WINExec branches are configured and packaged to run on OSX Sierra and Windows 10 in standalone. This means that it does not require any installation pre-requisites. 

Once downloaded, go to the TSP folder and run the following commands to execute the web application on the local server.

OSX
```bash
./manage runserver
```

A HTTP address should show and copy and paste it into the browser.

If you want to run the folder using the dependencies and developement environment we recommend that you use a virtual environment. The project is coded in Python 2.7.15. Using the correct technology version is important.

Using pip navigate to a the base folder the project is in:

```bash
pip install virtualenv
virtualenv arbitrary_folder_name
source arbitrary_folder_name/bin/activate
```

Go into the TSP folder where manage.py is:

```bash
cd TSP/bbsite
pip install -r requirements.txt
python manage.py runserver
```
Note that on each operating system there are some settings that may need changing in Django.

## Deployment
This Web application is also deployed to the web using Heroku.

```bash
https://finalbb.herokuapp.com/
```

## License
[MIT](https://choosealicense.com/licenses/mit/)