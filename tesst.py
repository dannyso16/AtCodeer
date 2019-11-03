import sys
import requests

def main(s:str):
  # print(s)

  endpoint = r'/api/hash'
  payload = {'q':s}
  r = requests.get('http://challenge-server.code-check.io'+endpoint, params=payload)
  print(r.json()['hash'])



if __name__ == '__main__':
    main('fizz')
