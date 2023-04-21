FROM python

RUN pip install flask scipy matplotlib
COPY chi_square.py kolmogorov_smirnov.py ./
COPY templates templates
RUN mkdir static

# ENTRYPOINT ["python", "chi_square.py"]
ENTRYPOINT ["flask", "run"]
