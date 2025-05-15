
import { useState } from "react";
import { 
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle 
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { BUILDINGS_LIST } from "@/constants";

interface BuildingAccessFormProps {
  onSubmit: (buildingId: string) => void;
  isLoading: boolean;
}


export const BuildingAccessForm = ({ onSubmit, isLoading }: BuildingAccessFormProps) => {
  const [selectedBuilding, setSelectedBuilding] = useState<string>("");
  
  const handleSubmit = () => {
    if (selectedBuilding) {
      onSubmit(selectedBuilding);
    }
  };
  
  return (
    <Card className="w-full mx-auto">
      <CardHeader>
        <CardTitle className="text-xl">Check Building Accessibility</CardTitle>
        <CardDescription>
          Select a building to check public transport accessibility options
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="space-y-2">
            <Select value={selectedBuilding} onValueChange={setSelectedBuilding}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a building" />
              </SelectTrigger>
              <SelectContent>
                <SelectGroup>
                  {BUILDINGS_LIST.map((building) => (
                    <SelectItem key={building} value={building}>
                      {building}
                    </SelectItem>
                  ))}
                </SelectGroup>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardContent>
      <CardFooter>
        <Button 
          className="w-full" 
          onClick={handleSubmit} 
          disabled={!selectedBuilding || isLoading}
        >
          {isLoading ? "Calculating..." : "Check Accessibility"}
        </Button>
      </CardFooter>
    </Card>
  );
};
