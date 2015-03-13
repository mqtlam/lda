%% Latent Dirichlet Allocation Demo

clear;

%% Constants
DOCWORD_PATHS{1} = 'data/docword.nips.txt';
DOCWORD_PATHS{2} = 'data/docword.kos.txt';
VOCAB_PATHS{1} = 'data/vocab.nips.txt';
VOCAB_PATHS{2} = 'data/vocab.kos.txt';
DATASET_NAME{1} = 'NIPS';
DATASET_NAME{2} = 'KOS';

%% Specify burn-in period
BURN_IN_PERIOD = 100; % iterations

%% Specify number of topics
LIST_OF_NUM_TOPICS = [2 5 10];

%% Run for each topic
for dataset = 1:2 % 1 = NIPS, 2 = KOS

    %% Read in data
    vocabWords = textread(VOCAB_PATHS{dataset}, '%s', 'delimiter', '\n');
    counts = dlmread(DOCWORD_PATHS{dataset}, ' ');
    nDocs = counts(1, 1);
    nVocabWords = counts(2, 1);
    counts = counts(4:end, :);
    
    % create corpus data structure: list of words (doc id, word id)
    nTotalWords = sum(counts(:, 3));
    corpus = zeros(nTotalWords, 2);
    curr = 1;
    for i = 1:length(counts)
        corpus(curr:curr+counts(i, 3)-1, :) = repmat(counts(i, 1:2), [counts(i, 3) 1]);
        curr = curr+counts(i, 3);
    end
    
    %% Run for each topic
    for nTopics = LIST_OF_NUM_TOPICS
        state_file = sprintf('state_dataset%d_topics%d.mat', dataset, nTopics);
        
        %% Priors (uniform)
        alpha = ones(nTopics, 1);
        beta = ones(nVocabWords, 1);
        
        %% Inference burn in period
        
        % set up initial counts and assignments
        if exist(state_file, 'file')
            % load previous state if interrupted
            load(state_file);
            beginning_iter = iter + 1;
        else
            % randomly initialize topic assignment and counts
            
            % current topic assignment for each of the words in corpus
            topicAssignments = randi(nTopics, [nTotalWords 1]);
            
            % number of words assigned to topic k in document d
            N_dk = zeros([nDocs nTopics]);
            
            % number of times word w is assigned to topic k
            N_kw = zeros([nTopics nVocabWords]);
            
            % total number of times any word is assigned to topic k
            N_k = zeros([nTopics 1]);
            
            % populate counts
            for i = 1:nTotalWords
                doc = corpus(i, 1);
                word = corpus(i, 2);
                topic = topicAssignments(i);
                
                N_dk(doc, topic) = N_dk(doc, topic) + 1;
                N_kw(topic, word) = N_kw(topic, word) + 1;
                N_k(topic) = N_k(topic) + 1;
            end
            
            beginning_iter = 1;
        end
        
        % iterate for burn-in period, then last iteration for sampling
        for iter = beginning_iter:BURN_IN_PERIOD+1
            fprintf('iter %d...', iter);
            tic;
            for i = 1:length(topicAssignments)
                % variables
                doc = corpus(i, 1);
                word = corpus(i, 2);
                topic = topicAssignments(i);

                % decrement counts: remove current assignment
                N_dk(doc, topic) = max(N_dk(doc, topic)-1, 0);
                N_kw(topic, word) = max(N_kw(topic, word)-1, 0);
                N_k(topic) = max(N_k(topic)-1, 0);

                % compute probability of word assigned to the topics
                P = zeros([nTopics 1]);
                for k = 1:nTopics
                    P(k) = (N_dk(doc, k) + alpha(k))...
                        * (N_kw(k,word) + beta(word))/(N_k(k) + beta(word) * nVocabWords);
                end
                P = P./sum(P);

                % sample from distribution above to assign topic
                topic = find(mnrnd(1, P'));
                topicAssignments(i) = topic;

                % update counts
                N_dk(doc, topic) = N_dk(doc, topic) + 1;
                N_kw(topic, word) = N_kw(topic, word) + 1;
                N_k(topic) = N_k(topic) + 1;
            end
            
            % save progress in case need to interrupt
            if mod(iter, 5) == 0
                save(state_file, 'topicAssignments', 'N_dk', 'N_kw', 'N_k', 'iter');
            end
            
            fprintf(' (%fs)\n', toc);
        end

        %% After burn-in: compute topic-word distribution
        psi = zeros(nTopics, nVocabWords);
        for k = 1:nTopics
            for w = 1:nVocabWords
                psi(k, w) = (N_kw(k, w) + beta(w))...
                    / (N_k(k) + nVocabWords*beta(w));
            end
            psi(k, :) = psi(k, :)./sum(psi(k, :));
        end
        
        %% Save variables
        save(sprintf('final_dataset%d_topics%d.mat', dataset, nTopics), 'N_dk', 'N_kw', 'N_k', 'psi');
        
        %% Print out top 10 most likely words and probabilities from each topic
        for k = 1:nTopics
           psi_k = [psi(k, :); 1:nVocabWords];
           psi_k = fliplr(sortrows(psi_k', 1)');

           fprintf('%s, Topic %d/%d:\n', DATASET_NAME{dataset}, k, nTopics);
           for i = 1:10
               fprintf('\t%s %f\n', vocabWords{psi_k(2, i)}, psi_k(1, i));
           end
        end
        
        fprintf('\n==============\n\n');
    end

end