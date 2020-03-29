    def generateTomIncompleteDataSet(self,hyp,cfg):
        #generates psi
        print("Generating Psi")
        self.psi=generateRandomQubit(dim)

        #generates phis, generates one less than whats needed
        #then duplicates one of the existing states
        #this ensures tomographic incompleteness
        print("Generating Phis")
        self.basis=generateRandomSet(dim,dim+1)
        a=[i for i in self.basis]
        a.append(self.basis[0])
        self.basis=np.array(a)
        

        #generates observations
        print("Generating Observations")
        for phi in self.basis:
            measurement=measureState(self.psi,phi)
            self.observations.append(measurement)
        self.observations=np.array(self.observations)
        
        #generates omegas
        print("Generating Questions")
        for i in range(0,cfg.dataSetLen):
            #generates omegas
            self.questions.append(generateRandomQubit(dim))


        #generates answers
        print("Generating Answers")
        for question in self.questions:
            measurement=measureState(self.psi,question)
            self.answers.append(np.array(measurement))
        self.answers=np.array(self.answers)


        #maps omegas onto parameterized omegas (done after finding soltuions to simplify math)
        print("Parameterizing Questions")
        self.omegaBasis=generateRandomSet(dim,dim+2)
        for i,omega in enumerate(self.questions):
            self.questions[i]=parameterizeQuestion(omega,self.omegaBasis)
        self.questions=np.array(self.questions,dtype=float)
        
        self.writeToFolder(cfg,"Incomplete")