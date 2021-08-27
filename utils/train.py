from sklearn.metrics import accuracy_score
import torch_geometric

def graph_accuracy(outs, y):return accuracy_score(outs.round().detach().numpy(),y)
def average(l): return sum(l)/len(l)


def train_one_epoch(model, train_items, val_items, graph_criterion, termination_criterion, optimizer, epoch, metric=graph_accuracy, device="cpu"):
    
    ### TRAINING
    model.train()
    for item in train_items:
        
        ## clear gradient
        optimizer.zero_grad()
        
        graph = item[0][0].to(device) ## first graph is first input
        true_graph = item[1][0].to(device) ## second graph is first output
        termination = item[1][1].to(device) ## termination state of the first input
        
        y,h,t = model(graph.x.float(),graph.edge_index) ## first prediction
        
        graph_loss = graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination) ## combine graph nodes and termination losses
        graph_loss.backward() ## backpropagate
        
        graph=true_graph ## next input is the previous output (in case of teacher forcing)
        
        for true_graph, termination in item[2:]: 
            true_graph, termination = true_graph.to(device), termination.to(device)
            
            h = h.detach() ## detach the inputs since they're outputs of the previous model computation step
            y = y.detach() ## detaching also prevents overfitting situations
            
            y,h,t = model(y,graph.edge_index,h) # model computation step
            #y,h,t = model(graph.x,graph.edge_index,h) ## teacher forcing
            
            graph_loss = graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination) ## compute loss
            graph_loss.backward() ## backpropagate 
            
            graph=true_graph ## next input is the previous output (in case of teacher forcing)
            
            optimizer.step() ## update weights once we're done going through all the steps 
        
        
    ### EVALUATION
    model.eval()
    loss_list = []
    mean_metric_list = []
    last_step_metric_list = []
    
    for item in val_items:
        
        graph = item[0][0].to(device) ## first graph is the first input
        true_graph = item[1][0].to(device) ## second graph is first output
        termination = item[1][1].to(device) ## termination state of the first input
        
        y,h,t = model(graph.x.float(),graph.edge_index) ## first prediction
        
        loss_list.append((graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination)).item())
        mean_metric_list.append(metric(y,true_graph.x))
        
        for true_graph, termination in item[2:]:  
            true_graph, termination = true_graph.to(device), termination.to(device)
            
            y,h,t = model(y,graph.edge_index,h)
            loss_list.append((graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination)).item())
            mean_metric_list.append(metric(y,true_graph.x))
        
        last_step_metric_list.append(metric(y,true_graph.x))
    
    loss, mean_metric, last_step_metric = average(loss_list), average(mean_metric_list), average(last_step_metric_list)
    print(f"Epoch {epoch} Loss {round(loss,2)} Mean step accuracy/Last step accuracy: {round(100*mean_metric,2)}/{round(100*last_step_metric,2)}")
    return loss, mean_metric, last_step_metric



def test(model, test_items, graph_criterion, termination_criterion, metric=graph_accuracy, device="cpu"):
    model.eval()
    loss_list = []
    mean_metric_list = []
    last_step_metric_list = []
    
    for item in test_items:
        graph = item[0][0].to(device)
        true_graph = item[1][0].to(device)
        termination = item[1][1].to(device)
        
        y,h,t = model(graph.x.float(),torch_geometric.utils.add_self_loops(graph.edge_index)[0])
        
        loss_list.append((graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination)).item())
        mean_metric_list.append(metric(y,true_graph.x))
        for true_graph, termination in item[2:]: 
            true_graph, termination = true_graph.to(device), termination.to(device)
            
            y,h,t = model(y,graph.edge_index,h)
            loss_list.append((graph_criterion(y,true_graph.x.float()) + termination_criterion(t,termination)).item())
            mean_metric_list.append(metric(y,true_graph.x))
        
        last_step_metric_list.append(metric(y,true_graph.x))
    
    loss, mean_metric, last_step_metric = average(loss_list), average(mean_metric_list), average(last_step_metric_list)
    print(f"TEST Loss {round(loss,2)} Mean step accuracy/Last step accuracy: {round(100*mean_metric,2)}/{round(100*last_step_metric,2)}")
    return loss, mean_metric, last_step_metric