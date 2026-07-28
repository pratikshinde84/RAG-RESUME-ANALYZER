M=[]
a=int(input("Enter a number you want to register"))
for i in range(0,a):
    num=int(input(f"Enter the number of book borrow by {i+1} member :"))
    M.append(num)
print("Number of book borrow by member",M)

L=[]
b=int(input("Enter a number 0f book register in library"))
for i in range(0,b):
    num1=int(input(f"How many type of book borrow by {i+1} membeHow many type of book borrow by 2 member : r :"))
    L.append(num1)
print("Number of book register in library",L)

count=0
sum=0
for i in range(0,a):
    if M[i]==0:
        count=count+1
    else:
        sum=sum+M[i]
avg=sum/a
print("Average no of book borrow by all member:",avg)
print("No of member who not borrow any book:",count)


high=L[0] 
for i in range(0,b):
    if L[i]>=high: 
        high=L[i]
print("Book with highest no of borrowing:",high)

low=L[0]
for i in range(0,b):
    if L[i]<=low:
        low=L[i]
print("Book with lowest no of borrowing:",low)       

count=[]
for i in range(0,high+1):
    cnt=1
    for j in range(0,b):
        if L[j]==i:
            cnt=cnt+1
    count.append(cnt)
print("Count is :",count)

high=0
for k in count:
    if k>=high:
        high=k
print("Higher count:",high)

fc=0
for z in range (0,high):
    if count[z]==high:
        fc=z
print("Higher frequency count is:",fc)