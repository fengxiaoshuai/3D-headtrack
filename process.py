from multiprocessing import Process, Queue

def f(q):
    q.put([42, None, 'hello'])

if __name__ == '__main__':
    q = Queue()
    try:
        p = Process(target=f, args=(q,))
        p.start()
        print(q.get())    # prints "[42, None, 'hello']"
        p.join()
    except Exception as e:
        print("Error:", e)