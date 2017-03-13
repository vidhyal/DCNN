package.path = package.path ..";".. paths.cwd().."/dataload/?.lua"
--print(package.path)
dl = require 'dataload'

function loadData(dataPath)

dataPath = dataPath or "/home/vidh/ILSVRC/Data/DET/"
--print(dataPath)
train, val = dl.loadImageNet(dataPath,1)
print("train")
--[[for key,value in pairs(train) do
   print("found member ".. key.. " ");
   --print (value)
end ]]--
--[[for key, value in pairs(train.dataset) do
--print(train.dataset)
   print( key .."     "..type(value))
end]]--

indices = torch.LongTensor():range(1, train.dataset.nsample)
inputs, targets = train.dataset:index(indices)



print("val")

indV = torch.LongTensor():range(1, val.dataset.nsample)
inV, tarV = val.dataset:index(indV)
return inputs, targets, inV, tarV
end

loadData()

function train(model, inputs, targets)
	

end











--[[for i = 1, train.dataset.nsample do
   print(i)
   indices = torch.LongTensor{i}
	inputs, targets = train.dataset:index(indices)
	print(inputs:size())
    print(targets)
   io.read()
end]]--




--print( train.dataset:getImageBuffer(1))


--print (type(train.dataset.imageClass))
--print(train.classinfo[932])
--print(train.classinfo[1855])
--print(train.dataset.classList[1])


--print(train.dataset.classListSample[2])
--print (train.dataset.imageClass)
--[[print(train.dataset.imgBuffers)
print (train.dataset)]]--
--print(val.dataset.imageClass)
--[[local count = 0;
for i,v in pairs(val.dataset.iclasses) do
	if  (train.dataset.iclasses[i] ==nil) then
        print(i.. "    ".. v)
		io.read()
		count = count +1
	end
end
print(count)]]--