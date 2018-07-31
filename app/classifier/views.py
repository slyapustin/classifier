from django.shortcuts import redirect
from django.views.generic import TemplateView

from classifier.predict import predict_category
from classifier.train import train


class IndexView(TemplateView):
    template_name = 'classifier/index.html'

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        text = self.request.GET.get('text')
        if text:
            context['text'] = text
            context['category'] = predict_category(text)

        return context


class TrainView(TemplateView):
    def get(self, request, *args, **kwargs):
        train()
        return redirect('classifier:index')
