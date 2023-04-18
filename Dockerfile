FROM python

RUN pip install flask scipy matplotlib
COPY . .

ENTRYPOINT ["python", "chi_square.py"]
