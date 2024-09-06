# Router by Embedding Models
This repo contains a code that uses embedding model to enroute the prompt to appropriate STATIC route. 


# Always run the code inside the virtual enviornment

# Study Metholodgy for 4 Configurations

(2,5) :  Route 2 | Uttarances per Route 5

(4,10) :  Route 4 | Uttarances per Route 10

(6,15) :  Route 6 | Uttarances per Route 15

(8,20) :  Route 8 | Uttarances per Route 20

![image](https://github.com/user-attachments/assets/f7e20323-e61b-40e8-a664-320fb2a30b91)

![image](https://github.com/user-attachments/assets/8d980a78-f93d-4cc9-852c-d95ec603b3d9)


![image](https://github.com/user-attachments/assets/a61f18ad-d471-4ba9-93c9-2e8d68200c77)


# Configuration Simialrity is Important (Use Same Configuration for both Server and Client)



* Server: static_router_2_5.py   <-------  Client: curl_caller_2_5.py
* Server: static_router_4_10.py  <-------  Client: curl_caller_4_10.py
* Server: static_router_6_15.py  <------- Client: curl_caller_6_15.py
* Server: static_router_8_20.py  <-------  Client: curl_caller_8_20.py

The result after each test should be saved into a excel file.
    
# Run Server
Firstly, run the server in one terminal.

```bash
$ python3 static_router_2_5.py
```

# Test with API Caller

Secodnly API call the server of same configuration

```bash
$ python3 curl_caller_2.5.py
```


