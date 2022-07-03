# Semantic Triplets for Algorithmic Close Reading
This repository corresponds to the article [Algorithmic Close Reading: Using Semantic Triplets to Index and Analyze Agency in Holocaust Testimonies](http://www.digitalhumanities.org/dhq/vol/16/3/000623/000623.html) and provides the code structure for using rule-based NLP methods to extract semantic triplets from Holocaust testimonies.

To cite our work, please use either of the the following ways:

* BibTex:

```python
@article{fan_presner_2022, title={Algorithmic Close Reading: Using Semantic Triplets to Index and Analyze Agency in Holocaust Testimonies}, volume={16}, number={3}, journal={Digital Humanities Quarterly}, author={Fan, Lizhou and Presner, Todd}, year={2022}} 
```

* APA7

> Fan, L., &amp; Presner, T. (2022). Algorithmic Close Reading: Using Semantic Triplets to Index and Analyze Agency in Holocaust Testimonies. Digital Humanities Quarterly, 16(3). 

## Download and try our process.
To start, call the `export_triplets_with_meta` function in `tri_main.py`.
Several test cases are commented out for your reference.
If you have any further question, please contact us at lizhouf@umich.edu or presner@ucla.edu

## Examples
Below, we provide two sample texts that can be used to run the semantic triplet extraction process. The CSV file in the [data folder](https://github.com/lizhouf/semantic_triplets/blob/main/data/) are the raw outputs, without any correction or curation. The manual annotations (column H) are provided to indicate clarifications, shortcomings, and/or corrections. 

Note: the coreference and object-based clustering are beta versions, which are not included in the Example outputs.

**EXAMPLE 1: Excerpt of Erika Jacoby’s testimony from the USC Shoah Foundation Visual History Archive** (Interview 8, recorded 1994). [514 words, 52 triplets extracted]

The next morning, we got out of the camp. And we went into town. And we looked for a house that was abandoned. And we found a very beautiful home, and we occupied it. We were 15 of us together from the same city. And my mother was the oldest. She was kind of the leader. And we got in there, and we couldn't lock the door, but we learned how to barricade it at night. And then I went out with one of my cousins, or one of my friends, to look for food. And the town was deserted. And the stores, of course-- there were all kinds of prisoners out looking for food. And they broke into the stores. 
And we broke into one of the bakeries. And I shoved a lot of bread into my-- we had a canvas bag of some sort, or a burlap bag that I got. And then we passed by a butcher store. And there were pigs hanging on the hook. And I took off half a pig, and carried it on my back back to the house. 
And my mother saw it, and she said good. We now have what to eat. And she cooked the pig. We religious Jews who never ate anything, and we ate it. And of course, we all got sick. And of course, we thought it was punishment for eating it. But we still had what to eat for a few days. Eventually, my mother had to go out and work for some Germans, some farmers, so that we could survive. Some of us went out to eat. And I was trying to find a way to get back to Hungary. We stayed in this town for many, many weeks. Now, one incident that I want to mention to you. Unfortunately I didn't bring it over here. I have a memento from this area. I broke into a house. That was like the following day. I think it was the house of the owner of the factory where I worked. 
And by this time, I had so much anger in me. This was probably the only time that I expressed that anger. I went into that house with an ax. I broke the piano and furniture and the Dutch masters on the wall. Whatever I could, I destroyed. Now, many people went into homes to rob. I did not want anything, except for three things that I took from the house. I took a white tablecloth. I took an apron. And I took a little silver cup that I still have that had the initials EF on it. And I decided that I would pretend that that belonged to my grandfather whose name was Engel Farkas. And I took that with me, and I kept it. And I really didn't understand. I was 16 by now, but I didn't understand that this was-- that I longed to establish normal life again, and these were the symbols of my home-- a white tablecloth, an apron, and a kiddush cup. So emotional. 



**EXAMPLE 2: Excerpt from Henry Coleman’s testimony from the USC Shoah Foundation Visual History Archive** (Interview 5371, recorded 1995). [882 words, 90 triplets extracted]. The transcript below has been lightly edited for punctuation (but English grammar has not been corrected). The excerpt below includes two portions around his arrival in Auschwitz. 

But then, in November 1942, they emptied the whole ghetto. But first, they separated young people and older people so we were in separate wagons. So we traveled a few days, maybe two days and a night, and we arrived in Auschwitz. 
And right there, you know, they used to separate you, women separate, young men separate, old people separate, women with children separate. So I was trying to get together with my sisters, but they wouldn't naturally. I was hit in the head, and I was put back in with the young people. 
And we marched from the railroad station. We marched to the camp. And my sisters were taken away, and I didn't see where. And all the other people were taken away also, to Birkenau, because the railroad station was in between Auschwitz and Birkenau. 
We came to Birkenau. They put us up in a big place. Then they took us into shower rooms. They shaved us, they took all our clothes away, naturally. They put us up with a kind of stinky liquid, you know. It was disinfectant. And then they tattooed our numbers. 

In fact, when they did the tattooing, it hurt, you know, because they go with needles. At first, it's bloody. Then it becomes blue, the way it is now. But I just twitched my head, and I was hit in the head by a hand. But it was very hard, because I moved my head, you know. 
After everything was done, they put us up into different blocks, different buildings. We were put into 7A with a group of young people, all young people, which was the upper part of the building. It was 7 and 7A. So we were on the top, on the second floor. 
The head of the block was a German of Polish descent, who was named Alfred Olshevsky. And he was screaming and yelling and kicking and beating. And each one was assigned to a bed. They were triple beds. And they showed us where the washroom is. And he told us, everybody has to be clean. Everyone has to wash himself, and he don't want anybody caught with dirty hands or dirty fingers or dirty feet or dirty ears or dirty head. 
And then they put us-- the next morning, they put us up in a room on the top of the building. It was like an attic. And they taught us to be bricklayers. First, they taught us about cement, about sand, about mixtures, about other things. Then they showed us how to lie the bricks down.
We were there for about two months. It was a very cold winter. And we were sitting there. Now, the food was very minimal, no food. The bread was like sawdust. 
And then every morning was roll call, and every evening was roll call. They called your numbers, not your names. And you had to answer to your number that you're there. 
And after two months they put us up on construction. I approached a man in the camp who was already months before me there. It was an older man. He was not old, because I was young, so I considered him old. He was my oldest brother's age. 
And he was a carpenter. I knew he worked outside the camp, Auschwitz camp. And he had the possibility of acquiring a little more food. So I approached him, and I asked him if he could help me with anything in food. 
And he said he cannot, but if I know anybody who has gold coins or dollars or German marks, he will be able to help me with a loaf of bread or with a piece of salami. Well, I immediately went into work,and I start asking around. And I said, whatever I'll get, we'll split. 
When we came into the camp, everything was taken. All the clothes was taken away from us. But the shoes, they let us take. So some people, for some reason, had either gold coins or money in their heels, buried in their heels. 
And the first night, he gave me a loaf of bread, which was a big thing. So I shared. I gave him half, and I took half. And I took the other half. I shared my half with another young man from my hometown, who was sleeping above me. 
In the evening, after roll call, my number was called out. And I really didn't know why. But I already got scared. And I was called into that political section of the Gestapo. And they start asking me, who gave me the money and who did I give the money to? 
Well, I already heard so many stories about what's going on in camp, so I figured, you know, I'm not going to involve anybody else anymore, that, you know, this is going to be my end, and that's it. And I resigned to the idea that that's it. So I wouldn't involve anyone. I got beat. They were beating me. 
They were hanging me by my hands, in the back, for about a half an hour, for 20 minutes. And then they were kicking me with bayonets. In fact, I have a big mark, here, on my stomach.

