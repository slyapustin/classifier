from django.views.generic import TemplateView

from classifier.models import Train
from classifier.predict import predict_category


class IndexView(TemplateView):
    template_name = 'classifier/index.html'

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        text = self.request.GET.get('text')
        if text:
            context['text'] = text
            context['category'] = predict_category(text)

        context['train'] = Train.objects.filter(finished__isnull=False).order_by('finished').first()

        return context
