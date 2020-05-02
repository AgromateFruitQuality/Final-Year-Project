import requests
import simplejson as json # to install simplejson ---> pip install simplejson==3.16.0


class ShelfLife:
    def __init__( self,city,fruit,colour,ripness=1,defect=0 ):  # Rr=30,firmness=55
        self.shelflife=0
        self.ripness=ripness
        self.colour=colour
        self.city=city
        self.fruit=fruit
        self.defect=defect


    def shelflife_claculator( self, past_shelf_days=0 ):
        self.findTemp()

        x=(self.colour * self.std_firm(self.fruit,self.ripness) )/( self.std_res(self.fruit) * self.temp * (self.defect+0.000001)) #adding 0.000001 to defect to avoid divide by zero error
        sl=0.004*(x**4) - 0.087*(x**3) + 0.679*(x**2) - 0.396*x + 0.839
        final_sl=sl - past_shelf_days
        if final_sl<=0:
            self.shelflife=-1
        else:
            self.shelflife=final_sl


    def getShelfLife( self ):
        return self.shelflife


    def std_res( self,fruit ):
        #standard respiration
        ft=fruit.lower()
        res = {"apple":45.7,
               "orange":52.8,
                "banana":47.0,
                "onion":52.8,
                "potato":21.6,
                "tomato":34.9  }
        return res[ft]


    def std_firm( self,fruit,i ):
        #standard firmness

        ft=fruit.lower()
        firm={"apple":[92,65,40],
            "orange":[90,79,50],
        "banana":[80,40,30],
        "onion":[100,80,70],
        "potato":[96,78,50],
        "tomato":[70,40,30] }
        return firm[ft][i]


    def findTemp( self ): #geeks for geeks
        # Enter your API key here
        api_key="10e2fe939efdce8c5e6f9845b11e0ba7" #from bhupesh's account in openwheathermapl.org
        # base_url variable to store url
        base_url="http://api.openweathermap.org/data/2.5/weather?"
        # complete_url variable to store
        # complete url address
        city_name=self.city
        complete_url=base_url+"appid="+api_key+"&q="+city_name
        # get method of requests module
        # return response object
        response=requests.get(complete_url)
        x=response.json()
        if x["cod"] != "404":
            y=x["main"]

            current_temperature=y["temp"]
            self.temp=int(current_temperature)-273

            current_pressure=y["pressure"]

            current_humidiy=y["humidity"]

            z=x["weather"]

            weather_description=z[0]["description"]

            # print following values
            print(" Temperature (in celcius unit) = ",self.temp)
            print("atmospheric pressure (in hPa unit) = "+
                  str(current_pressure)+
                  "\n humidity (in percentage) = "+
                  str(current_humidiy)+
                  "\n description = "+
                  str(weather_description))
        else:
            print(" City Not Found ")





a= ShelfLife("vadodara","Tomato",100,1,0.5) ## ShelfLife(city,fruit/veg name,color index,ripness,defect) 
                                            ## scale defect percentage(it must be a non-zero value) to a range of 0 to 1 before passing in the argunemt(by dividing it by 100). 
a.shelflife_claculator()  # shelflife_calculator(past_shelf_life [ 0 by default])
if a.getShelfLife()<0:
    print("Not edible !! Do not eat this !!")
else:
    print("shelflife = ",a.getShelfLife()," days.")

    #ripeness ----> 0 == HIGH firm fruit
    # ripeness ----> 1 == MID firm fruit
    # ripeness ----> 2 == LESS firm fruit


